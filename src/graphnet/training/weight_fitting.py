"""Classes for fitting per-event weights for training."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, List, Callable, Dict, Sequence, Union

import numpy as np
import pandas as pd
import sqlite3

from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
from tqdm import tqdm

from graphnet.data.utilities.sqlite_utilities import (
    create_table_and_save_to_sql,
)
from graphnet.utilities.logging import Logger


def _sqlite_table_exists(database_path: str, table_name: str) -> bool:
    """Return True if `table_name` exists in the SQLite database."""
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name = ?"
    with sqlite3.connect(database_path) as con:
        row = con.execute(query, (table_name,)).fetchone()
    return row is not None


def _sqlite_quote_identifier(name: str) -> str:
    """Safely quote a SQLite identifier."""
    return '"' + name.replace('"', '""') + '"'


def _save_table_checked(
    df: pd.DataFrame,
    table_name: str,
    database_path: str,
    force: bool = False,
) -> None:
    """Save DataFrame to SQLite with optional overwrite behavior."""
    if _sqlite_table_exists(database_path, table_name):
        if not force:
            raise ValueError(
                f"Table '{table_name}' already exists in '{database_path}'. "
                "Set force=True to overwrite it."
            )
        drop_sql = f"DROP TABLE {_sqlite_quote_identifier(table_name)}"
        with sqlite3.connect(database_path) as con:
            con.execute(drop_sql)
            con.commit()

    create_table_and_save_to_sql(df, table_name, database_path)


class WeightFitter(ABC, Logger):
    """Produces per-event weights.

    Weights are returned by the public method `fit_weights()`, and the weights
    can be saved as a table in the database.
    """

    def __init__(
        self,
        database_path: str,
        truth_table: str = "truth",
        index_column: str = "event_no",
    ):
        """Construct `UniformWeightFitter`."""
        self._database_path = database_path
        self._truth_table = truth_table
        self._index_column = index_column

        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__)

    def _get_truth(
        self, variable: str, selection: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Return truth `variable`, optionally only for `selection` events."""
        if selection is None:
            query = f"select {self._index_column}, {variable} from {self._truth_table}"  # noqa: E501
        else:
            query = f"select {self._index_column}, {variable} from {self._truth_table} where {self._index_column} in {str(tuple(selection))}"  # noqa: E501
        with sqlite3.connect(self._database_path) as con:
            data = pd.read_sql(query, con)
        return data

    def fit(
        self,
        bins: np.ndarray,
        variable: str,
        weight_name: Optional[str] = None,
        add_to_database: bool = False,
        force: bool = False,
        selection: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        db_count_norm: Optional[int] = None,
        automatic_log_bins: bool = False,
        max_weight: Optional[float] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fit weights.

        Calls private `_fit_weights` method. Output is returned as a
        pandas.DataFrame and optionally saved to sql.

        Args:
            bins: Desired bins used for fitting.
            variable: the name of the variable. Must match corresponding column
                name in the truth table.
            weight_name: Name of the weights.
            add_to_database: If True, the weights are saved to sql in a table
                named weight_name.
            force: If True and add_to_database=True, overwrite existing table
                with the same name. If False, raise ValueError when table exists.
            selection: a list of event_no's. If given, only events in the
                selection is used for fitting.
            transform: A callable method that transform the variable into a
                desired space. E.g. np.log10 for energy. If given, fitting will
                happen in this space.
            db_count_norm: If given, the total sum of the weights for the given
                db will be this number.
            automatic_log_bins: If True, the bins are generated as a log10
                space between the min and max of the variable.
            max_weight: If given, the weights are capped such that a single
                event weight cannot exceed this number times the sum of
                all weights.
            **kwargs: Additional arguments passed to `_fit_weights`.


        Returns:
            DataFrame that contains weights, event_nos.
        """
        # Member variables
        self._variable = variable
        self._add_to_database = add_to_database
        self._selection = selection
        self._bins = bins
        self._transform = transform
        if max_weight is not None:
            assert max_weight > 0 and max_weight < 1
            self._max_weight = max_weight
        else:
            self._max_weight = None

        if weight_name is None:
            self._weight_name = self._generate_weight_name()
        else:
            self._weight_name = weight_name

        truth = self._get_truth(self._variable, self._selection)
        if self._transform is not None:
            truth[self._variable] = self._transform(truth[self._variable])
        if automatic_log_bins:
            assert isinstance(bins, int)
            self._bins = np.logspace(
                np.log10(truth[self._variable].min()),
                np.log10(truth[self._variable].max() + 1),
                bins,
            )

        weights = self._fit_weights(truth, **kwargs)
        if self._max_weight is not None:
            weights[self._weight_name] = np.where(
                weights[self._weight_name]
                > weights[self._weight_name].sum() * self._max_weight,
                weights[self._weight_name].sum() * self._max_weight,
                weights[self._weight_name],
            )

        if db_count_norm is not None:
            weights[self._weight_name] = (
                weights[self._weight_name]
                * db_count_norm
                / weights[self._weight_name].sum()
            )
        if add_to_database:
            _save_table_checked(
                weights,
                self._weight_name,
                self._database_path,
                force=force,
            )
        return weights.sort_values(self._index_column).reset_index(drop=True)

    @abstractmethod
    def _fit_weights(self, truth: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def _generate_weight_name(self) -> str:
        pass


class Uniform(WeightFitter):
    """Produces per-event weights making variable distribution uniform."""

    def _fit_weights(self, truth: pd.DataFrame) -> pd.DataFrame:
        """Fit per-event weights.

        Args:
            truth: DataFrame containing the truth information.

        Returns:
            The fitted weights.
        """
        # Use quantile bins if requested, otherwise use provided bins
        if self._quantile_bins is not None:
            quantiles = np.linspace(0, 100, self._quantile_bins + 1)
            bins = np.unique(
                np.percentile(
                    truth[self._variable].dropna().values,
                    quantiles,
                )
            )
        else:
            bins = self._bins

        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=bins)

        # Get reweighting for each bin to achieve uniformity.
        # (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`, get the weight in
        # the corresponding bin
        ix = np.digitize(truth[self._variable], bins=bins) - 1

        sample_weights = np.zeros(len(truth), dtype=float)
        valid = (ix >= 0) & (ix < len(bin_weights))
        sample_weights[valid] = bin_weights[ix[valid]]

        # outside bins -> weight 0
        sample_weights[~valid] = 0

        sample_weights = sample_weights / sample_weights.mean()
        truth[self._weight_name] = sample_weights
        return truth.sort_values("event_no").reset_index(drop=True)

    def _gather_stats(self, truth: pd.DataFrame) -> pd.DataFrame:
        """Gather statistics about multiple databases in order to fit
        weights."""

    def _generate_weight_name(self) -> str:
        return self._variable + "_uniform_weight"

    def fit(
        self,
        variable: str,
        bins: Optional[np.ndarray] = None,
        quantile_bins: Optional[int] = None,
        add_to_database: bool = False,
        force: bool = False,
        db_count_norm: float = 1.0,
        automatic_log_bins: bool = True,
        max_weight: Optional[float] = None,
        weight_name: Optional[str] = None,
    ) -> pd.DataFrame:
        """Fit uniform weights.

        Args:
            variable: The variable to make uniform.
            bins: Bin edges to use. Ignored if quantile_bins is set.
            quantile_bins: If set, compute this many equal-count bins
                from the data quantiles, overriding `bins`. Guarantees
                no empty bins and avoids discontinuities in sparse
                high-energy regions.
            add_to_database: Whether to write weights back to the database.
            force: If True and add_to_database=True, overwrite existing table
                with the same name. If False, raise ValueError when table exists.
            db_count_norm: Target weighted event count for this database.
            automatic_log_bins: If True and bins is None, auto-generate
                log-spaced bins.
            max_weight: Maximum allowed weight as a fraction of total
                weight sum within this database.
            weight_name: Override the default weight column name.

        Returns:
            DataFrame with event_no and fitted weights.
        """
        assert (
            (bins is not None)
            or (quantile_bins is not None)
            or automatic_log_bins
        ), "Must provide one of: bins, quantile_bins, or automatic_log_bins=True"
        self._quantile_bins = quantile_bins
        return super().fit(
            variable=variable,
            bins=bins,
            add_to_database=add_to_database,
            force=force,
            db_count_norm=db_count_norm,
            automatic_log_bins=automatic_log_bins,
            max_weight=max_weight,
            weight_name=weight_name,
        )


class BjoernLow(WeightFitter):
    """Produces per-event weights.

    Events below x_low are weighted to be uniform, whereas events above x_low
    are weighted to follow a 1/(1+a*(x_low -x)) curve.
    """

    def _fit_weights(  # type: ignore[override]
        self,
        truth: pd.DataFrame,
        x_low: float,
        alpha: float = 0.05,
        percentile: bool = False,
    ) -> pd.DataFrame:
        """Fit per-event weights.

        Args:
            truth: DataFrame containing the truth information.
            x_low: The cut-off for the truth variable. Values at or below x_low
                will be weighted to be uniform. Values above will follow a
                1/(1+a*(x_low -x)) curve.
            alpha: A scalar factor that controls how fast the weights above
                x_low approaches zero. Larger means faster.
            percentile: If True, x_low is interpreted as a percentile of the
                truth variable.

        Returns:
            The fitted weights.
        """
        # Histogram `truth_values`
        bin_counts, _ = np.histogram(truth[self._variable], bins=self._bins)

        # Get reweighting for each bin to achieve uniformity.
        # (NB: No normalisation applied.)
        bin_weights = 1.0 / np.where(bin_counts == 0, np.nan, bin_counts)

        # For each sample in `truth_values`,
        # get the weight in the corresponding bin
        ix = np.digitize(truth[self._variable], bins=self._bins) - 1

        sample_weights = np.zeros(len(truth), dtype=float)
        valid = (ix >= 0) & (ix < len(bin_weights))
        sample_weights[valid] = bin_weights[ix[valid]]

        # outside bins -> weight 0
        sample_weights[~valid] = 0

        sample_weights = sample_weights / sample_weights.mean()
        truth[self._weight_name] = sample_weights  # *0.1
        bin_counts, _ = np.histogram(
            truth[self._variable],
            bins=self._bins,
            weights=truth[self._weight_name],
        )
        c = bin_counts.max()

        if percentile:
            assert 0 < x_low < 1
            x_low = np.quantile(truth[self._variable], x_low)

        slice = truth[self._variable][truth[self._variable] > x_low]
        truth[self._weight_name][truth[self._variable] > x_low] = 1 / (
            1 + alpha * (slice - x_low)
        )

        bin_counts, _ = np.histogram(
            truth[self._variable],
            bins=self._bins,
            weights=truth[self._weight_name],
        )
        d = bin_counts.max()
        truth[self._weight_name][truth[self._variable] > x_low] = (
            truth[self._weight_name][truth[self._variable] > x_low] * c / d
        )
        return truth.sort_values(self._index_column).reset_index(drop=True)

    def _generate_weight_name(self) -> str:
        return self._variable + "_bjoern_low_weight"


class PooledSplineCDFWeighter(Logger):
    """Fit event-by-event weights using a smoothed histogram/CDF spline in 1D.

    This class is intended for the log-space-uniform case where you want a
    smooth, monotone density estimate without KDE overhead.
    """

    def __init__(
        self,
        database_paths: Sequence[str],
        truth_table: str = "truth",
        index_column: str = "event_no",
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._database_paths = list(database_paths)
        self._truth_table = truth_table
        self._index_column = index_column

        self._features: Optional[List[str]] = None
        self._pdf_evaluators: Dict[str, Callable[[np.ndarray], np.ndarray]] = (
            {}
        )
        self._feature_ranges: Dict[str, tuple] = {}
        self._pooled_data: Dict[str, np.ndarray] = {}
        self._exclude_zeros: Dict[str, bool] = {}
        self._feature_transforms: Dict[str, Optional[Callable]] = {}

    def _read_truth(
        self, database_path: str, features: List[str]
    ) -> pd.DataFrame:
        feature_cols = ", ".join(features)
        query = f"select {self._index_column}, {feature_cols} from {self._truth_table}"
        with sqlite3.connect(database_path) as con:
            return pd.read_sql(query, con)

    def fit(
        self,
        features: Union[str, List[str]],
        percentile_range: tuple = (0.5, 99.5),
        exclude_zeros: Union[bool, Sequence[str]] = False,
        transform: Optional[
            Union[str, Callable, Dict[str, Optional[Union[str, Callable]]]]
        ] = None,
        spline_bins: int = 2048,
        spline_smoothing_sigma: float = 0.5,
    ) -> None:
        """Pool all databases and fit spline-smoothed histogram/CDF models."""
        if isinstance(features, str):
            features = [features]
        self._features = list(features)

        if isinstance(exclude_zeros, bool):
            self._exclude_zeros = {
                feature: exclude_zeros for feature in self._features
            }
        else:
            excluded = set(exclude_zeros)
            self._exclude_zeros = {
                feature: feature in excluded for feature in self._features
            }

        self._feature_transforms = self._resolve_feature_transforms(transform)
        self._pdf_evaluators = {}

        print(
            f"Pooling data from {len(self._database_paths)} databases for features: {self._features}",
            flush=True,
        )

        for feature in self._features:
            pooled_values = []
            for db in tqdm(
                self._database_paths, desc=f"  Reading {feature}", leave=False
            ):
                try:
                    df = self._read_truth(db, [feature])
                    values = df[feature].dropna().to_numpy(dtype=float)
                    if self._exclude_zeros.get(feature, False):
                        values = values[values != 0.0]
                    values = self._transform_values(values, feature)
                    values = values[np.isfinite(values)]
                    if len(values) > 0:
                        pooled_values.append(values)
                except Exception as exc:
                    print(
                        f"WARNING: Failed to read {feature} from {db}: {exc}",
                        flush=True,
                    )

            if not pooled_values:
                raise ValueError(
                    f"No valid data found for feature '{feature}'"
                )

            pooled_array = np.concatenate(pooled_values)
            pooled_array = pooled_array[np.isfinite(pooled_array)]
            if len(pooled_array) == 0:
                raise ValueError(f"No finite values for feature '{feature}'")

            self._pooled_data[feature] = pooled_array
            low_val = np.percentile(pooled_array, percentile_range[0])
            high_val = np.percentile(pooled_array, percentile_range[1])
            self._feature_ranges[feature] = (low_val, high_val)

            print(
                f"Feature '{feature}': {len(pooled_array)} values, range [{low_val:.4e}, {high_val:.4e}]",
                flush=True,
            )

            self._pdf_evaluators[feature] = (
                self._build_spline_hist_cdf_pdf_evaluator(
                    pooled_array,
                    spline_bins=spline_bins,
                    spline_smoothing_sigma=spline_smoothing_sigma,
                )
            )

    def _resolve_transform_spec(
        self, spec: Optional[Union[str, Callable]]
    ) -> Optional[Callable]:
        if spec is None:
            return None
        if callable(spec):
            return spec
        if isinstance(spec, str):
            s = spec.lower()
            if s in {"identity", "none"}:
                return None
            if s == "log10":
                return np.log10
            if s == "log":
                return np.log
        raise ValueError(
            "Invalid transform spec. Use one of: None, 'identity', 'log10', 'log', callable."
        )

    def _resolve_feature_transforms(
        self,
        transform: Optional[
            Union[str, Callable, Dict[str, Optional[Union[str, Callable]]]]
        ],
    ) -> Dict[str, Optional[Callable]]:
        assert self._features is not None
        if (
            transform is None
            or isinstance(transform, str)
            or callable(transform)
        ):
            fn = self._resolve_transform_spec(transform)
            return {feature: fn for feature in self._features}

        out: Dict[str, Optional[Callable]] = {}
        for feature in self._features:
            out[feature] = self._resolve_transform_spec(
                transform.get(feature, None)
            )
        return out

    def _transform_values(
        self, values: np.ndarray, feature: str
    ) -> np.ndarray:
        fn = self._feature_transforms.get(feature, None)
        x = np.asarray(values, dtype=float)
        if fn is None:
            return x
        with np.errstate(divide="ignore", invalid="ignore"):
            return fn(x)

    def _suppress_outliers(
        self,
        x: np.ndarray,
        feature: str,
        taper_width: float = 0.1,
    ) -> np.ndarray:
        low_val, high_val = self._feature_ranges[feature]
        range_val = high_val - low_val
        taper_abs = taper_width * range_val
        lower_center = low_val + taper_abs / 2
        upper_center = high_val - taper_abs / 2
        steepness = 6.0
        lower_taper = expit((x - lower_center) * steepness / taper_abs)
        upper_taper = expit(-(x - upper_center) * steepness / taper_abs)
        return lower_taper * upper_taper

    def _build_spline_hist_cdf_pdf_evaluator(
        self,
        values: np.ndarray,
        spline_bins: int,
        spline_smoothing_sigma: float,
    ) -> Callable[[np.ndarray], np.ndarray]:
        if spline_bins < 8:
            raise ValueError("spline_bins must be >= 8")
        if spline_smoothing_sigma < 0:
            raise ValueError("spline_smoothing_sigma must be >= 0")

        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            raise ValueError(
                "Need non-degenerate value range for spline_hist_cdf"
            )

        edges = np.linspace(vmin, vmax, spline_bins + 1, dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        counts = counts.astype(float)
        if spline_smoothing_sigma > 0:
            counts = gaussian_filter1d(
                counts, sigma=spline_smoothing_sigma, mode="nearest"
            )

        counts = np.maximum(counts, 1e-12)
        cdf_vals = np.concatenate([[0.0], np.cumsum(counts)])
        cdf_vals = cdf_vals / cdf_vals[-1]

        cdf_spline = PchipInterpolator(edges, cdf_vals, extrapolate=True)
        pdf_spline = cdf_spline.derivative()

        def _eval(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float).reshape(-1)
            x = np.clip(x, edges[0], edges[-1])
            pdf = pdf_spline(x)
            return np.maximum(pdf, 1e-12)

        return _eval

    def _compute_weights(
        self,
        x: np.ndarray,
        feature: str,
        suppress_outliers: bool = True,
        taper_width: float = 0.1,
    ) -> np.ndarray:
        if feature not in self._pdf_evaluators:
            raise ValueError(
                f"Feature '{feature}' not fitted. Call fit() first."
            )

        x = np.asarray(x, dtype=float)
        weights = np.zeros(len(x), dtype=float)

        valid = np.isfinite(x)
        if self._exclude_zeros.get(feature, False):
            valid = valid & (x != 0.0)

        if not np.any(valid):
            return weights

        x_eval = self._transform_values(x[valid], feature)
        finite_eval = np.isfinite(x_eval)
        if not np.any(finite_eval):
            return weights

        valid_ix = np.where(valid)[0][finite_eval]
        x_eval = x_eval[finite_eval]

        pdf_values = self._pdf_evaluators[feature](x_eval)
        valid_weights = 1.0 / np.maximum(pdf_values, 1e-10)

        if suppress_outliers:
            suppression = self._suppress_outliers(
                x_eval, feature, taper_width=taper_width
            )
            valid_weights = 1.0 + (valid_weights - 1.0) * suppression

        weights[valid_ix] = valid_weights
        return weights

    def apply(
        self,
        target_total_weight: float,
        suppress_outliers: bool = True,
        taper_width: float = 0.1,
        weight_suffix: str = "_uniform_weight",
        deploy: bool = False,
        force: bool = False,
        n_jobs: int = 1,
    ) -> Dict[str, pd.DataFrame]:
        if self._features is None:
            raise RuntimeError("Call fit() before apply().")

        print(
            f"Applying weights to {len(self._database_paths)} databases...",
            flush=True,
        )

        raw_results: Dict[str, Dict[str, Any]] = {}
        raw_weight_sums = {feature: [] for feature in self._features}

        def _compute_raw_result(db: str) -> tuple:
            try:
                df = self._read_truth(db, self._features)
                out_dict: Dict[str, Any] = {
                    self._index_column: df[self._index_column].to_numpy(),
                }
                for feature in self._features:
                    out_dict[feature] = df[feature].to_numpy()
                for feature in self._features:
                    x = df[feature].to_numpy(dtype=float)
                    w = self._compute_weights(
                        x,
                        feature,
                        suppress_outliers=suppress_outliers,
                        taper_width=taper_width,
                    )
                    out_dict[f"{feature}{weight_suffix}"] = w
                return db, out_dict
            except Exception as exc:
                print(
                    f"WARNING: Failed to apply weights to {db}: {exc}",
                    flush=True,
                )
                return db, None

        if n_jobs == 1:
            iterator = tqdm(
                self._database_paths,
                desc="  Computing raw weights",
                leave=False,
            )
            results = (_compute_raw_result(db) for db in iterator)
            for db, out_dict in results:
                if out_dict is None:
                    continue
                raw_results[db] = out_dict
                for feature in self._features:
                    raw_weight_sums[feature].append(
                        np.array(
                            [out_dict[f"{feature}{weight_suffix}"].sum()],
                            dtype=float,
                        )
                    )
        else:
            print(
                f"  Parallel preprocessing with {n_jobs} jobs...", flush=True
            )
            max_workers = None if n_jobs == -1 else n_jobs
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(_compute_raw_result, db)
                    for db in self._database_paths
                ]
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="  Computing raw weights",
                    leave=False,
                ):
                    db, out_dict = future.result()
                    if out_dict is None:
                        continue
                    raw_results[db] = out_dict
                    for feature in self._features:
                        raw_weight_sums[feature].append(
                            np.array(
                                [out_dict[f"{feature}{weight_suffix}"].sum()],
                                dtype=float,
                            )
                        )

        global_weights = {}
        for feature in self._features:
            if raw_weight_sums[feature]:
                global_weights[feature] = np.concatenate(
                    raw_weight_sums[feature]
                )

        total_raw_weight = np.mean([w.sum() for w in global_weights.values()])
        scale_factor = target_total_weight / total_raw_weight
        print(f"Weight scale factor: {scale_factor:.6f}", flush=True)

        out: Dict[str, pd.DataFrame] = {}
        for db, out_dict in raw_results.items():
            for feature in self._features:
                weight_name = f"{feature}{weight_suffix}"
                out_dict[weight_name] = out_dict[weight_name] * scale_factor

            wdf = (
                pd.DataFrame(out_dict)
                .sort_values(self._index_column)
                .reset_index(drop=True)
            )

            if deploy:
                for feature in self._features:
                    weight_name = f"{feature}{weight_suffix}"
                    deploy_df = wdf[[self._index_column, weight_name]]
                    _save_table_checked(
                        deploy_df,
                        weight_name,
                        db,
                        force=force,
                    )

            out[db] = wdf

        print("Weight application summary:", flush=True)
        for feature in self._features:
            total_weight = sum(
                out[db][f"{feature}{weight_suffix}"].sum() for db in out.keys()
            )
            print(
                f"  {feature}: total weight = {total_weight:.1f}", flush=True
            )

        return out
