"""
=============================================================================
ML 441/741  -  Assignment 1, Task 2
Assembles the Data Quality Issues & Handling Strategies table and writes it
to a submission-ready CSV.

WHAT THIS PRODUCES
------------------
    task2_dataQualityIssues.csv
        Columns, in the exact required order:
        Feature, Data Quality Issue, Evidence, Handling Strategy, Justification
        Saved with NO header row and NO index, and verified to survive
        pd.read_csv(<file>, header=None, names=<cols>).

WHY IT IS A SCRIPT
------------------
The rows are hand-written analysis, but keeping them in a script means you can
edit any cell, re-run, and always get a correctly formatted CSV back. Every
issue name is copied verbatim from dataQualityIssue.csv and every strategy from
handlingStrategy.csv, so nothing falls outside the allowed vocabulary.
=============================================================================
"""

import pandas as pd

COLS = ["Feature", "Data Quality Issue", "Evidence",
        "Handling Strategy", "Justification"]

# Each tuple is one row, in feature order (A1..A46 then T). A feature appears
# on multiple rows when it has multiple distinct issues (allowed by the spec).
rows = [
    # A1
    ("A1", "Outliers",
     "Min. = -1.26 (a negative value where the bulk sit near zero) and Max. = 60 against a 3rd Qrt. of only 0.69, so both extremes lie far from the centre.",
     "Clamp the instances",
     "Isolated values at both ends fall well beyond the interquartile range and would distort scale-sensitive estimates; clamping caps them at sensible bounds while keeping every row and the rest of the distribution intact."),

    #  A2
    ("A2", "Cardinality",
     "Card. = 133 distinct levels, by far the highest of any categorical feature (every other categorical has 12 or fewer); the two modes tcp (47.75%) and udp (35.98%) already cover about 84%, leaving a long tail of roughly 131 rare levels.",
     "Bin the feature into intervals",
     "The many sparsely populated levels add width and noise while carrying little signal; grouping the rare categories into a single 'other' level keeps the informative categories and brings the cardinality under control."),

    #  A3
    ("A3", "Missing Values",
     "% Miss. = 54.85, so more than half the column is absent (Card. = 12 among the present values).",
     "Remove the feature",
     "Above roughly half missing there is too little observed data to impute without inventing the majority of the column, so discarding the feature is safer than fabricating it."),

    #  A5
    ("A5", "Skewness",
     "Strong right skew: Mean 19.8 against Median 4, with values trailing out to Max. 10,646.",
     "Apply log transformation to the feature",
     "A long right tail pulls the summary statistics away from the bulk of the data; a logarithm reshapes the distribution toward symmetry so the typical values are no longer compressed against the axis."),
    ("A5", "Correlation",
     "The correlation with A15 is 0.97, very close to +1, which the notes describe as a very strong positive correlation, so the two features are effectively redundant.",
     "Remove one of the features",
     "A5 belongs to a tightly correlated group with A8 and A15; because these repeat the same variation, only one representative needs to be retained."),
    ("A5", "Correlation",
     "The correlation with A8 is 0.96, close to +1 (a very strong positive correlation), confirming the two carry almost the same information.",
     "Remove one of the features",
     "The near-linear relationship confirms A5 and A8 duplicate one another within the correlated group, so one is redundant."),

    #  A6
    ("A6", "Skewness",
     "Right-skewed: Mean 18.5 against Median 2, Max. 11,018.",
     "Apply log transformation to the feature",
     "The heavy upper tail distorts the feature's centre and spread; a log transform evens out the distribution and reduces the tail's dominance."),
    ("A6", "Correlation",
     "The correlation with A16 is 0.98, very close to +1 (a very strong positive correlation), so the two are redundant.",
     "Remove one of the features",
     "The strong correlation shows A6 and A16 largely encode the same information, making one of them redundant."),
    ("A6", "Correlation",
     "The correlation with A9 is 0.97, close to +1 (a very strong positive correlation).",
     "Remove one of the features",
     "A6, A9 and A16 form a mutually correlated group; since they repeat one another, only one should be kept."),

    #  A7
    ("A7", "Independent",
     "The histogram is flat: a uniform distribution on [0,1] with Std. Dev. = 0.2883 (the value expected of a uniform) and Card. = 257,673, one distinct value per row. The notes state that a uniform distribution may indicate an irrelevant feature, and this one shows no relationship with the target.",
     "Remove the feature",
     "A uniform, unique-per-row feature with no relationship to the target behaves like injected random noise, so it carries no information for prediction and risks being fitted as spurious structure."),

    #  A8
    ("A8", "Skewness",
     "Right-skewed: Mean 8,573 against Median 528, Max. 14,355,774.",
     "Apply log transformation to the feature",
     "With most values small and a very long tail, a logarithm compresses the tail and brings the distribution closer to symmetric, giving each value more balanced influence."),
    ("A8", "Correlation",
     "The correlation with A15 is 1.00 (0.996), essentially +1, so the two features are almost perfectly redundant.",
     "Remove one of the features",
     "The pair moves together almost perfectly, so one is redundant; dropping either retains the shared information without the wasted dimension."),

    #  A9
    ("A9", "Skewness",
     "Right-skewed: Mean 14,387 against Median 178, Max. 14,657,531.",
     "Apply log transformation to the feature",
     "The large gap between mean and median signals a long tail; a log scale draws in the extreme values so they no longer dominate the feature's behaviour."),
    ("A9", "Correlation",
     "The correlation with A16 is 1.00 (0.997), essentially +1, an almost perfect positive correlation.",
     "Remove one of the features",
     "Two near-perfectly correlated features duplicate the same signal; keeping one preserves the information while removing the redundancy."),

    #  A10
    ("A10", "Invalid Values",
     "1,386 rows (0.538%) hold the non-numeric token 'a' in an otherwise numeric feature. Because 'a' is not the '?' missing marker, these are invalid rather than absent, so A10's % Miss. stays 0.00 and the corruption surfaces only as invalid values.",
     "Replace values with missingness",
     "The letter is not a valid measurement and is almost certainly a data-entry or integration error; recoding it as missing removes the corruption while letting the small gap be handled as ordinary missingness."),

    #  A11
    ("A11", "Missing Values",
     "% Miss. = 0.51; the same 0.51% rate recurs across several unrelated features, pointing to the same rows being missing everywhere (a data-integration artefact).",
     "Impute the values with the median",
     "With only about half a percent absent, imputation barely perturbs the distribution, and the median is used because the feature is not symmetric, keeping the filled value representative."),
    ("A11", "Cardinality",
     "Only 13 distinct numeric values across 257,673 rows, bounded at 0 to 255 and clustered near the top.",
     "Bin the feature into intervals",
     "A numeric feature with so few distinct levels behaves more like an ordinal category than a continuous quantity; grouping it into intervals reflects that and avoids implying a false, fine-grained scale."),

    #  A12
    ("A12", "Invalid Values",
     "The bar plot shows one category of roughly 29,603,031 among otherwise byte-range values (0, 32, 252, 253, 254), orders of magnitude outside the feature's evident domain.",
     "Replace values with missingness",
     "A single value so far outside the feature's normal range is inconsistent with its domain and is best treated as an error, converting it to missing rather than trusting it."),

    #  A13
    ("A13", "Magnitude",
     "Values span about 1.2e24 to 5.99e29 with Mean 7.06e27, dwarfing every other feature by many orders of magnitude.",
     "Apply log transformation to the feature",
     "Magnitudes this large dominate any scale-sensitive calculation and risk numerical overflow; a logarithm compresses them onto a comparable scale while preserving their order."),

    #  A14
    ("A14", "Skewness",
     "Right-skewed: Mean 658,214 against Median 1,747, Max. 22,422,730.",
     "Apply log transformation to the feature",
     "A logarithm rebalances a distribution whose mean sits far above its median, reducing the leverage of the largest values."),

    #  A15
    ("A15", "Outliers",
     "Median 0 and 3rd Qrt. 3, yet Max. = 5,319, so a handful of values sit far above the mass at zero.",
     "Clamp the instances",
     "The extreme values are rare spikes far outside the typical range; capping them limits their disproportionate influence without discarding instances."),

    #  A16
    ("A16", "Outliers",
     "Median 0 and 3rd Qrt. 2, with Max. = 5,507.",
     "Clamp the instances",
     "A few points lie orders of magnitude above the central mass; clamping them to a bound keeps the feature usable while curbing their leverage."),

    #  A17
    ("A17", "Cardinality",
     "Card. = 1: the feature holds the single value 1 for all 257,673 instances (Mode % = 100).",
     "Remove the feature",
     "A feature with only one value has zero variance and cannot distinguish between instances, so it carries no predictive information and only adds an empty dimension."),

    #  A18
    ("A18", "Cardinality",
     "Card. = 1: the feature holds the single value 0 for all 257,673 instances (Mode % = 100).",
     "Remove the feature",
     "With a single constant value the feature cannot separate any classes and contributes nothing to prediction, so retaining it only increases dimensionality."),

    #  A19
    ("A19", "Skewness",
     "Right-skewed: Mean 912 against Median 0.38, Max. 84,371.",
     "Apply log transformation to the feature",
     "The distribution is concentrated near zero with a long tail; a log transform spreads the low values out and compresses the tail toward symmetry."),

    #  A20
    ("A20", "Outliers",
     "Median 0.007 and 3rd Qrt. 56, but Max. = 57,739, far beyond the upper quartile.",
     "Clamp the instances",
     "The maximum dwarfs the interquartile range, marking clear high outliers; a clamp caps these extremes so they do not dominate the feature's scale."),

    #  A21
    ("A21", "Skewness",
     "Right-skewed: Mean 5,419 against Median 0.67, Max. 1,483,831.",
     "Apply log transformation to the feature",
     "Because the bulk of the values are tiny relative to the maximum, a logarithm is needed to reveal the structure among the small values rather than letting the tail set the scale."),

    #  A22
    ("A22", "Missing Values",
     "% Miss. = 0.51 (Median = 0); the identical rate seen in other features suggests a shared set of missing rows.",
     "Impute the values with the median",
     "The missing fraction is tiny, and because the feature is heavily right-skewed the median (0) is a more representative fill than the mean, which the long tail would inflate."),
    ("A22", "Outliers",
     "Median 0 and 3rd Qrt. 120, yet Max. = 463,199.",
     "Clamp the instances",
     "The huge gap between the third quartile and the maximum identifies extreme high values; clamping bounds them while preserving all instances."),

    #  A23
    ("A23", "Correlation",
     "The correlation with A26 is 0.98 (a very strong positive correlation), and the two hold identical values on 99.01% of rows.",
     "Remove one of the features",
     "These are effectively the same feature recorded twice; keeping both would double-count the same signal, so one should be removed."),
    ("A23", "Cardinality",
     "Card. = 22, with values concentrated at 0 and 255 (a near-bimodal, byte-range feature).",
     "Bin the feature into intervals",
     "The handful of distinct, clustered values behave categorically rather than continuously, so binning into intervals represents them more faithfully."),

    #  A24
    ("A24", "Missing Values",
     "% Miss. = 0.51 on a feature reaching about 4.3e9; Median = 0.",
     "Impute the values with the median",
     "At half a percent the fill has negligible effect, and the median resists the enormous upper values that would make a mean-based fill unrepresentative."),
    ("A24", "Magnitude",
     "Typical values are enormous (3rd Qrt. about 2.0e9, Max. about 4.3e9) while most features sit between 0 and a few hundred.",
     "Apply normalisation to the feature",
     "When features differ by many orders of magnitude, the large-scale ones dominate distance and scale-based computations; normalising brings all features onto a shared range so each contributes comparably."),

    #  A25
    ("A25", "Missing Values",
     "% Miss. = 0.51 on a large-scale feature (values up to about 4.3e9); Median = 0.",
     "Impute the values with the median",
     "The absent fraction is negligible and the median avoids the distortion that the feature's extreme upper values would introduce into a mean."),
    ("A25", "Magnitude",
     "Values reach about 4.3e9 (3rd Qrt. about 2.0e9), far above the scale of most other features.",
     "Apply normalisation to the feature",
     "Rescaling this feature to a common range prevents its sheer magnitude from overwhelming features measured on much smaller scales."),

    # ---- A26 ----
    ("A26", "Cardinality",
     "Card. = 19, with values clustered at 0 and 255.",
     "Bin the feature into intervals",
     "Like A23, this feature takes only a few clustered values, so discretising into bins matches its categorical character better than a continuous scale."),

    # ---- A27 ----
    ("A27", "Correlation",
     "A27 is very strongly correlated with A28 (r = 0.94) and A29 (r = 0.92), both close to +1, so the three features move together as near-duplicates.",
     "Remove one of the features",
     "A27, A28 and A29 form a tightly correlated group that carries nearly the same variation, so only one representative needs to be retained and the others are redundant."),

    # ---- A28 ----
    ("A28", "Missing Values",
     "% Miss. = 0.51 (Median = 0), matching the shared missingness rate across features.",
     "Impute the values with the median",
     "The small missing share and the feature's skew make the median the safer central-tendency fill, leaving the observed distribution essentially unchanged."),

    # ---- A29 ----
    ("A29", "Missing Values",
     "% Miss. = 0.51 (Median = 0).",
     "Impute the values with the median",
     "Imputing so few values with the median preserves the shape of this skewed feature while retaining every instance."),

    # ---- A30 ----
    ("A30", "Skewness",
     "The histogram is unimodal and right-skewed: Mean 137.6 sits well above Median 73, with values trailing out to Max. 1,504 against a 3rd Qrt. of only 100.",
     "Apply log transformation to the feature",
     "A right skew pulls the mean above the median and lets the upper tail dominate the feature; a logarithm draws the tail in and makes the distribution more symmetric so the typical values are better represented."),

    # ---- A31 ----
    ("A31", "Skewness",
     "The histogram is unimodal and right-skewed: Mean 121.6 against Median 44, trailing out to Max. 1,500 with a 3rd Qrt. of only 89.",
     "Apply log transformation to the feature",
     "The gap between mean and median together with the long upper tail is a clear right skew; a log transform compresses the tail and brings the bulk of the distribution toward symmetry."),

    # ---- A32 ----
    ("A32", "Missing Values",
     "% Miss. = 0.51 (Median = 0, values almost all zero).",
     "Impute the values with the median",
     "The tiny missing fraction and the near-zero centre mean a median fill is both representative and low-impact."),
    ("A32", "Cardinality",
     "Card. = 14, with almost all values 0 (Median = 3rd Qrt. = 0) and a small tail to 172.",
     "Bin the feature into intervals",
     "With so few distinct values dominated by zero, treating it as continuous overstates its resolution; binning groups the sparse non-zero values sensibly."),

    # ---- A33 ----
    ("A33", "Outliers",
     "Median 0 and 3rd Qrt. 0 (most values are zero), with Max. = 6,558,056.",
     "Clamp the instances",
     "Almost all values are zero while a sparse few are enormous, a textbook outlier pattern; capping the extremes stops them from distorting the feature."),

    # ---- A34 ----
    ("A34", "Correlation",
     "The correlation with A45 is 0.98, close to +1 (a very strong positive correlation).",
     "Remove one of the features",
     "The very high correlation shows A34 and A45 carry nearly the same signal, making one redundant."),
    ("A34", "Correlation",
     "The correlation with A39 is 0.95, a very strong positive correlation.",
     "Remove one of the features",
     "A34, A39 and A45 form a mutually correlated block of count features; only one representative needs to be retained."),

    # ---- A36 ----
    ("A36", "Correlation",
     "The correlation with A37 is 0.96, close to +1 (a very strong positive correlation).",
     "Remove one of the features",
     "The two count-like features track each other closely; keeping one avoids double-counting the same variation."),

    # ---- A38 ----
    ("A38", "Correlation",
     "A38 is very strongly correlated with A37 (r = 0.91), close to +1, so the two count features largely repeat the same information.",
     "Remove one of the features",
     "A38 sits in the correlated block of count features with A36 and A37; because it duplicates their variation, keeping a single representative is sufficient."),

    # ---- A39 ----
    ("A39", "Correlation",
     "The correlation with A45 is 0.96, a very strong positive correlation.",
     "Remove one of the features",
     "A39 and A45 sit in a correlated block of count features and largely repeat each other, so one can be dropped."),

    # ---- A40 ----
    ("A40", "No information",
     "Card. = 257,673, equal to the number of rows; Min. = 1, Max. = 257,673, Mean = Median = 128,837 = (N+1)/2, and the quartiles fall at N/4 and 3N/4, i.e. a perfect 1-to-N counter.",
     "Remove the feature",
     "The values are a running row index rather than a measured property, so they encode only record position and cannot generalise to unseen data."),
    ("A40", "Instance duplication",
     "After removing the unique index A40 and the noise feature A7, 84,544 of 257,673 rows (32.81%) are exact duplicates; A40's 1-to-N values are what otherwise make every row appear unique.",
     "Remove the instances",
     "Roughly a third of the records are repeats once the artificial index is set aside; duplicated rows over-weight those patterns and bias the learned distribution, so the redundant copies should be dropped."),

    # ---- A41 ----
    ("A41", "No information",
     "Mode 0 covers 98.74% of rows (Card. = 4), so the bar plot is dominated by a single level and the feature is almost constant, leaving virtually no variation between instances.",
     "Remove the feature",
     "One level dominates almost entirely, so the feature is effectively constant and offers no useful signal to separate instances."),

    # ---- A42 ----
    ("A42", "Correlation",
     "A42 equals A41 on 100% of the 256,358 shared rows; A41 is integer-coded (0, 1, 2, 4) and A42 is the float version (0.0, 1.0, 2.0, 4.0).",
     "Remove one of the features",
     "The columns are the identical feature stored in two data types, so one is pure duplication; removing either eliminates both the redundancy and the type inconsistency."),

    # ---- A43 ----
    ("A43", "Cardinality",
     "Card. = 11, with almost all values 0 (Median = 3rd Qrt. = 0) and Max. = 30.",
     "Bin the feature into intervals",
     "A low-cardinality, mostly-zero feature is better described by a few intervals than a continuous scale, which would imply detail the data does not contain."),

    # ---- A44 ----
    ("A44", "Correlation",
     "A44 is very strongly correlated with A37 (r = 0.91) and A36 (r = 0.90), both close to +1.",
     "Remove one of the features",
     "A44 belongs to the same correlated block of count features and repeats the signal already present in A36 and A37, so it is redundant and one of the group can be dropped."),

    # ---- A45 ----
    ("A45", "Missing Values",
     "% Miss. = 0.51 (Median = 4).",
     "Impute the values with the median",
     "With half a percent missing, a median fill leaves the distribution intact while keeping all rows available."),

    # ---- A46 ----
    ("A46", "No information",
     "Mode 0 covers 98.11% of rows (Card. = 3), so the bar plot is dominated by a single level, leaving almost no variation in the feature.",
     "Remove the feature",
     "The overwhelming dominance of a single level leaves almost no variation to learn from, so the feature does not help distinguish instances."),

    # ---- T ----
    ("T", "Missing Values",
     "% Miss. = 0.1 of the target label.",
     "Remove the instances",
     "A missing target cannot be imputed without fabricating ground truth, and at 0.1% the affected rows can be dropped with negligible loss."),
    ("T", "Skewness",
     "10 classes with a dominant majority: Mode '0.0' at 55.52% while the 2nd Mode '9.0' is only 11.19% and the remaining classes trail to about 1%.",
     "Oversample the minority class(es)",
     "A skewed class distribution biases learning toward the majority and can make a model look accurate while ignoring rare classes; oversampling the minority classes rebalances the training distribution so every class is represented."),
]

# Build, save, and verify the round-trip the spec asks for.
df = pd.DataFrame(rows, columns=COLS)
OUT = "task2_dataQualityIssues.csv"
df.to_csv(OUT, index=False, header=False)

check = pd.read_csv(OUT, header=None, names=COLS)
assert check.shape == df.shape, "round-trip shape mismatch"

# sanity: every issue and strategy must be inside the allowed vocabularies
ALLOWED_ISSUES = {
    "Cardinality", "Correlation", "Data inconsistency", "Independent",
    "Instance duplication", "Invalid Values", "Magnitude", "Missing Values",
    "No information", "Not robust", "Outliers", "Poor representation",
    "Sampling bias", "Skewness", "Temporal dependence"}
ALLOWED_STRAT = {
    "Add Gaussian noise", "Apply log transformation to the feature",
    "Apply normalisation to the feature", "Bin the feature into intervals",
    "Clamp the instances", "Create a new feature to indicate missingness",
    "Do nothing", "Encode the feature using integers",
    "Encode the features using one-hot encoding",
    "Impute the values using complete case analysis",
    "Impute the values with the mean", "Impute the values with the median",
    "Impute the values with the mode",
    "Map feature to 2 new features using sine and cosine transformations",
    "Oversample the minority class(es)", "Remove one of the features",
    "Remove the feature", "Remove the instances",
    "Replace values with missingness", "Rescale the feature",
    "Undersample the majority class(es)"}
bad_i = set(df["Data Quality Issue"]) - ALLOWED_ISSUES
bad_s = set(df["Handling Strategy"]) - ALLOWED_STRAT
assert not bad_i, f"issue names outside vocabulary: {bad_i}"
assert not bad_s, f"strategy names outside vocabulary: {bad_s}"

print(f"Wrote {OUT}: {len(df)} rows, columns verified, vocabulary verified.\n")
print("Rows per issue:")
print(df["Data Quality Issue"].value_counts().to_string())
print("\nRows per strategy:")
print(df["Handling Strategy"].value_counts().to_string())