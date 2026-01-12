What are the “advanced TFT features”?

    Variable Selection Networks (VSNs)
        For each input type (static, past, future), TFT learns soft attention over features: at each time step, it assigns a weight to each variable and mixes them into one vector.
        This is both:
            Performance-oriented: ignore noisy/useless features.
            Interpretability: you get per-feature importance over time (they show nice tables & plots of which variables matter).
        Why high impact:
            Directly handle the “lots of features” situation: OHLCV + TA + on-chain + sentiment.
            Provide strong interpretability: you can show which features matter more than others (great for thesis discussion).
            TFT ablation shows static covariate encoders and VSNs increase performance more than gating alone.

    Gated Residual Networks (GRNs) & GLU gating
        Every major block (variable selection, temporal processing, etc.) has a gated residual wrapper (GRN).
        The GLU gate can “turn down” or almost disable a block if it’s not useful for a dataset, so the model adapts its effective depth and complexity.
        Why:
            Ablation shows they improve performance, especially on smaller/noisier datasets. 
            But they don’t change your story as much as VSNs or future covariates.

    Known future inputs (future covariates)
        Time-varying features that are known in advance for each future step: calendar features (day-of-week, holiday flags), scheduled events, planned halving dates, etc. 
        TFT uses them in a decoder so it can predict multiple future steps while incorporating expected seasonality and events.
        Crypto TFT works often treat technical / on-chain indicators as “known covariates” when they are precomputable for the whole dataset window, even for future prediction days.
        Why:
            Cheaper than full encoder–decoder; you can start by just adding calendar embeddings (day-of-week, month) as additional inputs.
            Papers highlight the importance of handling known future covariates in multi-horizon settings, and calendar effects exist in crypto as well (day-of-week anomalies, etc.).


0. Overall plan & order
To keep you sane and avoid breaking everything at once, I’d suggest this order:
    Update config.py once to prepare for all three ideas (VSN, future inputs, gating), but in a minimal way.
    Add gating (GRNs / GLUs) to tft_model.py — this only touches the model and is conceptually local.
    Add Variable Selection Networks (VSNs) to tft_model.py (plus a few new things in config.py).
    Add “known future inputs” via:
        config.py: define which features are future covariates.
        data_pipeline.py: produce those features and pass them to the model.
        tft_model.py: extend forward signature to accept them and fuse them into the temporal block.
So: config base updates → gating → VSN → known future inputs.


1. Step 0 – Prepare config.py for advanced features
Here we don’t implement logic; we just add structure so the rest of the code has a clean place to plug into.
What to add conceptually
    Feature group lists
        Keep your existing FEATURE_COLS as “all features”.
        Add more specific groups:
            PAST_COVARIATE_COLS – features you observe in the past (what you already use now).
            FUTURE_COVARIATE_COLS – features known in advance (e.g. ["day_of_week", "month"] once we add them).
            (Optional for later) STATIC_COLS – features that don’t change over time (for multi-asset / regimes).
    ModelConfig flags
        In ModelConfig:
            use_gating: bool = True
            use_variable_selection: bool = True
            use_future_covariates: bool = False (we’ll flip this once we implement them)
            (Optional for future) use_static_covariates: bool = False
    VSN & GRN sizes
        Add a few hyperparameters:
            variable_selection_hidden_size: int = 64
            static_hidden_size: int = 16 (for future use)
            future_hidden_size: int = 32 (if you want separate dims for future covariates later)
Steps to do this
    Open config.py.
    Under your feature definitions, add the PAST / FUTURE lists. For now, you can set PAST_COVARIATE_COLS = FEATURE_COLS and FUTURE_COVARIATE_COLS = [].
    Extend ModelConfig with the booleans and extra sizes (no code in other files changes yet).
    Re-run your project (or just import config in a Python shell) to make sure nothing breaks.


2. Step 1 – Add gating (GRNs, GLUs) to tft_model.py
This is internal to the model and does not require any change to data.
Conceptual changes
    Right now your model roughly does:
        input_projection → LSTM → attention → feedforward → output.
    We’ll:
        Introduce a GRN module:
            GRN(x, context=None) → same shape as x
        with:
            a small MLP,
            a GLU gate,
            residual connection,
            layer norm.
        Wrap key transformations inside GRNs:
            Instead of a plain linear + ReLU for the FFN, use a GRN.
            Optionally wrap the input projection and/or attention output in GRNs.
Steps to do this
    Refactor tft_model.py into blocks mentally (no code yet):
        Input projection block.
        LSTM encoder block.
        Attention block.
        Feedforward block.
    Define a plan for GRNs:
        Decide: We’ll keep LSTM as-is for now and use GRNs for:
            Variable-selection (later),
            Post-attention transformation,
            Final FFN.
    Design the GRN interface:
        class GatedResidualNetwork(nn.Module):
            forward(x, context=None) → returns tensor same shape as x.
        Internally: linear → (add optional context) → ELU/ReLU → linear → GLU gate → residual + layer norm.
    Decide where to plug GRNs in the current flow:
        After attention: instead of attn_layer_norm(attn_out + enc), we can have:
            attn_processed = attn_grn(attn_out, context=None) with its own residual inside.
        For FFN: replace existing FFN with a GRN, or keep FFN but wrap it inside one.
    Implementation order when you actually code (later):
        Create the GatedResidualNetwork class.
        Replace one simple linear block with a GRN (e.g. your FFN).
        Run a dummy forward pass (in __main__) to ensure shapes are unchanged.
        Then optionally extend gating to other parts.
At this stage, once coded, you’d still call:
model(x) with the same input/output shapes as now.


3. Step 2 – Add Variable Selection Networks (VSNs)
Once gating is in place, we add VSNs, which are the most important “TFT-ish” bit for many features.
Conceptual changes
    Instead of:
        Taking all features at once and projecting them with input_projection,
    we will:
        Treat each feature as its own “channel”, transform each separately to a hidden vector, and then learn soft weights for each feature at each time step.
    VSN for past covariates will:
        Take x_past with shape (B, T, F_past) where F_past = len(PAST_COVARIATE_COLS).
        For each feature j:
            Pass x[..., j] (the scalar) through a small linear layer to embed it to dimension H.
        Compute a feature importance score for each j using a GRN + softmax.
        Combine feature embeddings with these weights to get a single vector per time step with dimension H.
    This replaces your current single input_projection.
config.py changes needed
    Make sure PAST_COVARIATE_COLS is correctly set (for now, same as FEATURE_COLS).
    Add ModelConfig.variable_selection_hidden_size if not already there.
    Optionally, add a flag use_variable_selection = True so you can turn it off easily.
tft_model.py changes conceptually
    Create a VariableSelectionNetwork module, which:
        Stores one small linear (or GRN) per feature.
        Has another GRN to compute attention scores over features.
        Returns:
            selected: tensor (B, T, H) — the result of mixing features at each time step.
            weights: tensor (B, T, F_past) — feature importance weights.
    Replace input_projection usage in forward:
        Instead of x_proj = input_projection(x), you call:
            x_selected, vsn_weights = vsn_past(x_past)
        x_selected then flows into LSTM as before.
    Decide what to do with vsn_weights:
        For now, you can just ignore them in the forward pass (or optionally return them when return_attention=True).
        Later, you can add an option to return them for plotting in your thesis.
    Maintain same overall shape into LSTM and rest of model:
        Ensure x_selected is (B, T, hidden_size).
    Data pipeline impact
        None at this stage, as long as your dataset already passes (B, T, F) with the same features.
        VSN just changes how we process those features inside the model.


4. Step 3 – Add simple “known future inputs”
   1. config.py – define future feature names & sizes
        1.1 Calendar + halving feature names
            Add new lists under your feature section:
                Base (non-shifted) calendar columns:
                    CALENDAR_COLS = ["day_of_week", "is_weekend", "month"]
                Halving base columns:
                    HALVING_COLS = ["is_halving_window", "days_to_next_halving"] (or just one of them)
            These are features attached to each date (t).
            Then define the future versions (what the model will use as known future):
                FUTURE_CALENDAR_COLS = ["dow_next", "is_weekend_next", "month_next"]
                FUTURE_HALVING_COLS = ["is_halving_window_next"]
            Finally:
                FUTURE_COVARIATE_COLS = FUTURE_CALENDAR_COLS + FUTURE_HALVING_COLS
        1.2 ModelConfig updates
            Your ModelConfig already has use_future_covariates and future_input_size
            Once the pipeline + model are wired:
                Set use_future_covariates = True
                Keep:
                    future_input_size = len(FUTURE_COVARIATE_COLS)
            This tells the model: “expect a future covariate vector of that length”.
            (We flip use_future_covariates only after the other steps compile.)
   2. data_pipeline.py – compute & attach future covariates 
         We add three conceptual pieces:
             Calendar base features
             Halving base features
             “Next-day” (future) versions and sequence building
         2.1 Add calendar base features
             Create a small helper
                 def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
                 # Use df.index (DatetimeIndex)
                 # Add: day_of_week, is_weekend, month
                 return df
             Call this after load_btc_daily and before target creation in prepare_datasets:
                 df = load_btc_daily(...)
                 df = add_technical_indicators(df)
                 df = add_calendar_features(df)
                 # df = add_halving_features(df)  # next step
                 df = add_target_column(df, up_threshold=config.UP_THRESHOLD)
         2.2 Add halving base features
             Create another helper:
                def add_halving_features(df: pd.DataFrame) -> pd.DataFrame:
                 # 1) define a list of known/approx halving dates as pd.Timestamp
                 # 2) for each date in df.index, compute:
                 #    - nearest upcoming halving
                 #    - days_to_next_halving
                 #    - is_halving_window (within ± N days window, e.g. 90)
                 return df
             Hard-code a small list of halving dates (2012, 2016, 2020, 2024, and maybe the next approx).
             For each row date:
                 find the smallest halving date >= that date,
                 days_to_next_halving = (next_halving - date).days (clip at some max if you want).
                 is_halving_window = int(abs((date - nearest_halving).days) <= window_size).
             Note: these columns do not need to be in FEATURE_COLS unless you also want to use them as past covariates. For now, they’re mainly ingredients for future covariates.
         2.3 Create “next-day” future covariates
             We want a separate set of columns that represent t+1 information attached to t. Two options:
                 Option A: compute them in a dedicated function,
                 Option B: fold them into add_target_column.
             Order matters:
                 Start from df with base calendar + halving + OHLCV + indicators.
                 Compute target and future covariates on the full df (before dropping). For example:
                     df = add_future_covariates(df)
                     df = add_target_column(df, ...) (where add_target_column already drops the last row with NaN future_return_1d).
                 Because we use shift(-1) for the future covariates, the last row will also have NaNs in _next columns. Dropping the last row in add_target_column will remove these NaNs at the same time.
             Result: for any remaining row t, we have:
                 target_up[t] = up/down from t → t+1
                 dow_next[t], is_weekend_next[t], month_next[t], is_halving_window_next[t] = info about t+1
         2.4 Adjust sequence creation to return future covariates
             Currently build_sequences only returns:
                 sequences (N, seq_length, n_features)
                 labels (N,)
             We want an extended version that also returns future vectors:
                 future_covariates (N, n_future_features)
             Plan:
                 Compute a future_array once per split
                 In the sliding-window loop, for each end_idx:
                     seq_x = past features from start_idx: end_idx+1 (as before).
                     y = target_array[end_idx] (as before).
                     f = future_array[end_idx] (this row already encodes t+1 via shift(-1)).
                 Append f into a list and finally np.stack to future_covariates.
             You can:
                 Either create a new function build_sequences_with_future(...) returning sequences, future_covariates, labels,
                 Or update build_sequences to optionally return future covariates when a future_cols argument is provided.
         2.5 Extend BTCTFTDataset to hold future inputs
             Right now BTCTFTDataset takes just sequences, labels and __getitem__ returns (X, y).
             We want:
                 Constructor accepts either:
                     only past_sequences and labels, or
                     past_sequences, future_covariates, labels.
                 If future_covariates is provided, store it as a tensor and make __getitem__ return (X_past, X_future, y).
             This can be done in a backwards-compatible way with a flag or by checking if future_covariates is None.
   3. tft_model.py – use future inputs in the model
     Finally, we teach the model to actually use x_future.
     3.1 Change the forward signature: We’ll conceptually change it to
             def forward(self,
             x_past: torch.Tensor,
             x_future: torch.Tensor | None = None,
             return_attention: bool = False):
         x_past shape: (batch_size, seq_length, input_size) (same as current x).
         x_future shape: (batch_size, future_input_size) (one vector per sample, describing tomorrow).
         When you later adapt train_tft.py, each batch will be
             for x_past, x_future, y in dataloader:
             logits = model(x_past, x_future)
         When use_future_covariates=False, you can keep compatibility by allowing x_future=None and ignoring it.
     3.2 Add a “future covariate encoder”
         In __init__, after the existing blocks, add:
             A small encoder for x_future, e.g.:
                 Either a simple linear:
                     self.future_projection = nn.Linear(
                         self.config.future_input_size, hidden_size
                     )
             Or a GRN, consistent with the rest of your architecture:
                     self.future_grn = GatedResidualNetwork(
                         input_size=self.config.future_input_size,
                         hidden_size=self.config.variable_selection_hidden_size,
                         output_size=hidden_size,
                         dropout=dropout,
                     )
             We’ll use this to map (B, future_input_size) → (B, hidden_size).
     3.3 Fuse future context with temporal representation
         In your current forward, you eventually compute something like:
             ff_out – the temporal representation after LSTM + attention + temporal GRN.
             last_timestep = ff_out[:, -1, :] – summary of past 30 days.
         Then you do:
             logits = self.output_layer(last_timestep)
         We’ll change this part:
             After last_timestep is computed, and if use_future_covariates is True:
                 future_context = self.future_grn(x_future) or self.future_projection(x_future)
                 → shape (B, hidden_size).
             Fuse last_timestep and future_context, e.g. with another GRN:
                 Option A (context GRN):
                     fused = self.decision_grn(last_timestep, context=future_context)
                     logits = self.output_layer(fused)
                 Option B (concat + linear):
                     combined = torch.cat([last_timestep, future_context], dim=-1)
                     fused = some_linear_or_grn(combined)
                     logits = self.output_layer(fused)
             If use_future_covariates is False (or x_future is None):
                 Fall back to the current behavior:
                     logits = self.output_layer(last_timestep).
         This keeps:
             The temporal encoding path unchanged for past covariates (good for comparing with/without future inputs).
             The future info entering right at the “decision stage”, which matches the idea “we know tomorrow’s calendar & halving context when making the prediction”.


5. Summary: what you actually do, step-by-step (no code yet)
Config groundwork
    Add feature groups (PAST_COVARIATE_COLS, FUTURE_COVARIATE_COLS, etc.).
    Add model flags (use_gating, use_variable_selection, use_future_covariates).
    Add a few new sizes (variable_selection_hidden_size, etc.).
Gating (GRNs / GLUs) – only tft_model.py
    Introduce a GatedResidualNetwork.
    Wrap your FFN (and maybe post-attention) with it.
    Keep the external model API the same.
Variable Selection Networks – tft_model.py (+ small config)
    Design a VariableSelectionNetwork that:
        Takes (B, T, F) past covariates.
        Outputs (B, T, H) + feature weights (B, T, F).
    Replace input_projection with this.
Known future inputs – all three files
    config.py: define calendar/future feature names & sizes; set use_future_covariates=True when ready.
    data_pipeline.py: create calendar features, build x_future for each sample, extend dataset to return them.
    tft_model.py: update forward to accept x_future, encode & fuse it with final temporal representation via concat or GRN context.