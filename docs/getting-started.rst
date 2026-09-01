getting started
===============

This is where you describe how to get set up on a clean install, including the
commands necessary to get the raw data (using the `sync_data_from_s3` command,
for example), and then how to make the cleaned, final data sets.

help_vs_auto.py data flow
==========================

``analysis/model/help_vs_auto.py`` reads its data columns without defining
them in the file itself -- they come from an earlier pipeline stage. This is
a quick reference for where each column originates and which functions expect
it, so the module doesn't have to be read top-to-bottom to make sense of it.

Pipeline order (also the actual execution order, driven by ``run()`` at the
bottom of the file):

1. ``build_events(cfg)`` (in ``analysis/features/help_vs_auto_features.py``)
   creates ``events_df`` with columns ``subject``, ``trial``, ``run``,
   ``step``, ``label``, ``window``, ``sge``, ``gte``.

   - ``label``: 1 = manual alt-press, 0 = automatic recommendation
     (``event_timeline()``, lines 32-57 of that file).

2. ``build_features(cfg, events_df)`` computes additional per-event gaze
   features and is merged onto ``events_df`` (on ``subject``, ``trial``,
   ``run``, ``step``, ``label``, ``window``) to form ``features_df``.

3. ``prep_window(df, w, features)`` (in ``help_vs_auto.py``) slices
   ``features_df`` to one time window and adds two derived columns used only
   within this module:

   - ``subject_cat``: ``subject`` cast to a pandas category, used as the
     random-effect grouping key for leave-one-subject-out (LOSO) splits.
   - ``fixed_effect_cols``: the list of z-scored feature names (e.g.
     ``sge_z``, ``gte_z``), used as the fixed effects in the GLMM formula and
     as the feature columns for the Random Forest.

4. ``balanced_sample(df, seed)`` downsamples to equal counts per ``label``,
   producing ``balanced_df``, which is what ``loso_glmm`` and ``loso_sklearn``
   actually train/test on.

Quick lookup:

======================  ========================================  =========================================
Column/variable          Created in                                 Meaning
======================  ========================================  =========================================
``label``                ``help_vs_auto_features.py`` (events)      1 = manual alt-press, 0 = automatic
``window``                ``help_vs_auto_features.py`` (events)      Look-back window length, in seconds
``subject_cat``           ``help_vs_auto.py: prep_window``           Categorical subject ID (random effect)
``fixed_effect_cols``     ``help_vs_auto.py: prep_window``           Z-scored feature names (fixed effects)
``balanced_df``           ``help_vs_auto.py: balanced_sample``       Class-balanced slice used for LOSO fit/test
======================  ========================================  =========================================

Glossary: functions
--------------------

Plain-English description of every function in ``help_vs_auto.py``, in the
order they appear in the file:

``balanced_sample(df, seed)``
    Randomly drops rows so both ``label`` values (0 and 1) have the same
    count. Prevents the model from just learning "predict the majority
    class."

``loso_glmm(balanced_df, fixed_effect_cols)``
    Fits a mixed-effects logistic regression (GLMM), holding out one
    subject's data at a time (leave-one-subject-out) and predicting on it.
    Returns the true labels and predicted probabilities collected across all
    subjects.

``loso_sklearn(balanced_df, fixed_effect_cols, make_model)``
    Same leave-one-subject-out idea as ``loso_glmm``, but for any
    scikit-learn-style classifier (here, Random Forest) instead of the GLMM.

``score(y_true, y_pred, threshold)``
    Turns predicted probabilities into a class prediction (using
    ``threshold``) and computes ROC-AUC and accuracy.

``make_rf(hva_cfg, n_estimators_key)``
    Builds a ``RandomForestClassifier`` with settings pulled from the config
    dict.

``prep_window(df, w, features)``
    Filters the data down to one time window ``w``, drops rows with missing
    features, and adds ``subject_cat`` and the z-scored ``fixed_effect_cols``.
    Shared setup step used by every function below.

``run_classifiers(cfg, df)``
    For each time window, repeatedly balances the data, runs both the GLMM
    and Random Forest through leave-one-subject-out, and averages their
    AUC/accuracy across repeats. Returns one summary row per window.

``confusion_matrices(cfg, df)``
    Same loop as ``run_classifiers``, but accumulates confusion matrices
    (true/false positives and negatives) instead of AUC/accuracy, and derives
    recall/precision/specificity from them.

``feature_importance(cfg, df)``
    Fits one Random Forest per window on balanced data and measures
    permutation importance: how much AUC drops when each feature is shuffled.

``run(cfg)``
    The orchestrator. Builds the events and features tables, saves them to
    CSV, then calls ``run_classifiers``, ``feature_importance``, and
    ``confusion_matrices`` in turn and saves their outputs too. This is the
    entry point -- everything else in the file exists to support this.

Glossary: variables
--------------------

Short-lived variable names that show up inside the functions above:

``n_min``
    The size of the smaller of the two ``label`` groups; how many rows
    ``balanced_sample`` keeps from each group.

``rng``
    A seeded random number generator, so sampling is reproducible.

``formula``
    The Patsy/R-style regression formula string, e.g.
    ``"label ~ sge_z + gte_z"`` -- passed to the GLMM to say "predict
    ``label`` from these features."

``y_true`` / ``y_pred``
    Parallel lists built up across all leave-one-subject-out folds: the
    actual labels and the model's predicted probabilities.

``s``
    One subject ID, while looping over subjects for leave-one-subject-out.

``train`` / ``test``
    The data split for one leave-one-subject-out fold: every subject except
    ``s`` (``train``), and just subject ``s`` (``test``).

``model`` / ``fit``
    The (unfit) model object and the fitted result after calling
    ``.fit_vb()`` or ``.fit()``.

``test_exog``
    The held-out fold's predictor matrix, built by hand (intercept column
    plus each ``fixed_effect_cols`` column) because ``fit.predict()`` for the
    GLMM needs a raw numeric matrix rather than a DataFrame.

``p`` / ``pred``
    ``p``: predicted probabilities (0-1). ``pred``: those probabilities
    turned into 0/1 class predictions by comparing to ``threshold``.

``threshold``
    The probability cutoff above which a prediction counts as class 1
    (manual alt-press).

``hva_cfg``
    Shorthand for ``cfg["help_vs_auto"]`` -- the config block with settings
    specific to this analysis (feature list, number of repeats, etc).

``features``
    The list of raw (not yet z-scored) feature column names to use, read
    from ``hva_cfg["features"]``.

``n_repeats`` / ``seed0``
    How many times to redraw a balanced sample per window, and the base
    random seed for those draws.

``base``
    The window-filtered, feature-complete slice of the data, before class
    balancing (output of ``prep_window``).

``w``
    One time window value in seconds (e.g. 5, 10, 15, 20).

``scores`` / ``cms``
    Accumulators: ``scores`` collects AUC/accuracy per repeat;
    ``cms`` collects summed confusion matrices, one per model type
    (``"glmm"``, ``"rf"``).

``row`` / ``rows``
    A dict (or list of dicts) of results for one window, later turned into a
    DataFrame.

``aucs`` / ``accs``
    Lists of AUC and accuracy values collected across the repeated balanced
    draws for one window.

``tn`` / ``fp`` / ``fn`` / ``tp``
    True negatives, false positives, false negatives, true positives -- the
    four cells of a confusion matrix.

``X`` / ``y``
    Standard scikit-learn convention: ``X`` is the feature matrix, ``y`` is
    the target (``label``) column.

``rf``
    A fitted ``RandomForestClassifier`` instance.

``perm``
    The result of ``permutation_importance()``: how much a model's score
    drops when one feature's values are shuffled, per feature.

``processed``
    The folder path where input/output CSVs live, from
    ``cfg["paths"]["processed"]``.

``events_df`` / ``features_df``
    The two main tables built in ``run()``: raw labeled events, and events
    joined with computed gaze features.

``results`` / ``importance`` / ``cm``
    The three output DataFrames from ``run()``: classifier scores per
    window, permutation feature importances, and confusion matrices.
