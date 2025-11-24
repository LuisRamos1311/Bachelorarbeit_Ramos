# Bachelorarbeit_Ramos
Bachelorarbeit Wirtschaftsinformatik


Data aquired from
https://www.cryptodatadownload.com/data/gemini/
first work will be will daily data from 2015 to 2025-11-18

Installed talib from https://ta-lib.org/install/?utm_source=chatgpt.com#executable-installer-recommended

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
    Under your feature definitions, add the PAST / FUTURE lists. 
    For now, you can set PAST_COVARIATE_COLS = FEATURE_COLS and FUTURE_COVARIATE_COLS = [].
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





Created data_pipeline
Test prints:
dataset sizes
shape of the first sample
its label
first 3 timesteps of that sequence
detailed last timestep with feature names
it additionally prints the last 3 timesteps of the same sequence
This just helps you see how the window starts and how it ends 
(how indicators and prices evolve over the 30 days). 
It doesn’t change any data, it’s only for inspection and understanding.

Trying to expand the project to add tools that differentiate tft from others
GRNs, VSNs, known future inputs (this couldve probably have been done in the end)