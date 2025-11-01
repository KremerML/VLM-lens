1. Setup & Exploration
   - Load LLaVa, load color-object binding dataset
   - Run model, collect baseline accuracy
   - Identify where binding happens:
     * Activation patching across layers
     * Attention pattern analysis (vision → text tokens)
     * Logit attribution for correct color predictions

2. Target Layer Selection
   - Based on step 1, select 3-5 layer ranges:
     * Late vision encoder layers (if using full model)
     * Vision-language connector region
     * Early-to-mid language model layers
   - Hypothesis: Binding likely happens in connector + early LM layers

3. CLT Training
   - For each target layer range:
     * Collect activations on diverse dataset (not just binding task)
     * Train CLT with appropriate hyperparameters
     * Validate reconstruction quality
   - Save trained transcoders

4. Feature Analysis
   - For each CLT:
     * Identify features that activate for colors (visual)
     * Identify features that activate for color words (text)
     * Find "binding features" that activate for BOTH
   - Automated labeling: max-activating examples

5. Circuit Discovery
   - Path tracing: How do "red pixel" features → "red word" features?
   - Ablation: Remove specific features, measure task degradation
   - Intervention: Can you swap "red" → "blue" by manipulating features?
   - Build circuit diagram showing feature flow

6. Validation
   - Test circuit on held-out examples
   - Try adversarial cases (unusual colors, misleading text)
   - Compare to baseline methods (attention rollout, SAEs)