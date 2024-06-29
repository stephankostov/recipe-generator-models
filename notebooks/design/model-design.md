## Problem Scope

Our task at hand here is to generate recipes, which has two core requirements: to generate food tokens, and their respective quantities. 

## Splitting Models 

For these two requirements we need to either design a model which is able to predict both, or split the task up and have two models for each requirement.

### Split Models

Ultimately this is the more simple approach, for two core reasons:

- Implementation: It's more straightforward to split the task up and train specific models for each.
- Maintainability: Models can be retrained individually when required.

### Singular Model

Ideally this would be what to strive for as it will be able to capture the dependancy between tokens and their quantities the generation of each.

However the implementation is difficult here. How can we encode this information of token_id and quantity into the model, to be able to encode and decode both. 

### Decision

For simplicity we have opted to first implement the split-model approach to begin with.

## Transformer Architecture

Conveniently, what makes sense here is a Transformer architecture. The model needs to understand information of its context ie. the previous ingredients presented to it. This understanding includes an encoding of what that ingredient is, as well as the relationships between them. The encoding here is already done through our data (molecular composition), it's the relationships between each in the context that we require from the architecture, which is precisely what the attention mechanism is for. In particular this mechanism is able to capture relationships of many levels of complexity (multi-heads), and by its nature, entirely independent of their positonal distance.

The idea behind this project was heavily inspired by LLM's and my desire to work with them, so it makes sense that this ends up being the same problem framing as them.

## Molecular Composition Encoding

Why can't we just use an LLM assistant to do this? Practically, we simply can and it will work great, but this project is really for practice building generative models so this was the niche. With this meaningful encoding there is also the potential for adding much greater interpretability of the model's decision making, to possibly provide insights on what makes a good recipe. 

Now how can we incorporate this molecular composition data into the model? As the numerical vectorised data that the model requires as input, we can simply feed in our tokens as vectors corresponding to the (scaled) concentration for each sample (ie. ingredient).

### Position Embeddings

Do we remove the positional embeddings? Yes, as all these do is encode information about the position of the tokens, which in our case is irrelevant. 

Point of confusion: doesn't the prediction require the positions to be known in order to make predictions at the correct position? No, this is not how this works. At each position, the decoder makes predictions depending on what it has seen before it, through masking.

## Types of Tansformers

The transformer architecture is an obvious choice, but from here which type of transformer would be best? There are two main decisions to make when condisidering this: [https://huggingface.co/transformers/v3.1.0/model_summary.html]

- encoder, decoder, or both?
- autoregressive or non-autoregressive?

### Encoder-Decoder vs Decoder-only

The intuition behind this decision is whether or not we need an additional block in order to understand the input data seperate from that which is generating the output data. ie. Do we expect our input data to be different from our output data?

#### Ingredient Tokens

Lets take for our simplest case of generating the ingredient tokens. Our input data is simply just a set of previous tokens, and the outputs are those with the addition of the newly generated tokens. The data is fundamentally the same, so we only require a decoder for this.

#### Ingredient Quantities

This will involve a sequence of ingredient tokens as input, and output of their respective quantites. In this case, we will need an encoder to encode the ingredient tokens, and a decoder which will be able to generate numeric quantites.

### Autogressive vs Non-Autoregressive

Do we want the model to generate tokens one at a time, or do we want the model to output the completed sequence all at once? Note this this is referring to the generation mechanism at *inference* time. 

Do we want the model to iteratively generate to the final output, or do we want it to generate it all at once. Treating the model as an all-knowing black-box we would think that it should be able to generate the right answer at once. When viewing the model's generation as thinking, intuitively, a model should be more accurate if allowed to predict each token with consideration to all others in the sequence - those prior as well as the ones after. In other words, it should be able to change its predictions of tokens beforehand depending on those it chooses later.

But this view of the model thinking could very well be flawed. Rather than thinking it's moreso doing a one-step calculation to come up with everything at once. Maybe allowing it to iterate is more close to thinking as we know it, as thought isn't a one-step process of neurons, it's many over time which evolves and forms the final product. 

Intuition here can only go so far and gets complicated fast. Empirically it is known that for token generation, autoregressive generation gives more accurate results, at the expense of having more of an inference cost as inference is done for each token[^1]. 

Since this application is very constrained - to recipes with 1k ingredient tokens rather than text about anything known to the internet - the model can be small, and therefor inference will not be costly so should not be a concern. We are much more interested in accuracy, and therefor when knowing this fact autoregressive models are an easy choice.

#### Ingredient Model

Autoregressive as stated above.

#### Quantity Model

When it comes to quantity prediction though, it makes sense to be able to predict every quantity at once, as the quantities are done in proportion to the whole recipe.  

This might be flawed when training though, as there may be many different weight combinations of a recipe. Take a cake for example, there are many different concentrations of flour/sugar/butter/water corresponding to different types of cakes. If this is the case the model would learn to generate mean values for each.

#### Quantity Dependancy

Could this be a flaw in the method here - can the outputs of each token be dependent on eachother like this?

The way the data is set up, as proportions of the total recipe, we must generate the full result at once because to the loss can only be calculated once the full recipe has been generated. 

We could remove this dependancy, and get the model to generate weights instead, in an autoregressive manner.

# Inference Sampling

There are a number of options to consider when designing the inference sampler. Do we want a standard sampler which just samples from the probability distribution generated from the model's activations? This is what GPT uses, and is known to give a varying amount of results. Do we want to implement a more engineered method such as beam search, which performs a search of possible sequences from the top $k$ most probable tokens, and selects the sequence with the highest total probability.


---

[^1]: "Non-autoregressive machine translation models can significantly improve decoding speed by predicting every word in parallel (Gu et al., 2018; Libovicky ́ & Helcl, 2018). This advantage comes at a cost to performance since modeling word order is trickier when the model cannot condition on its previous predictions." [https://arxiv.org/abs/1905.11006]

[^2]: "Non-autoregressive machine translation models significantly speed up decoding by allowing for parallel prediction of the entire target sequence. However, modeling word order is more challenging due to the lack of autoregressive factors in the model." [https://proceedings.mlr.press/v119/ghazvininejad20a/ghazvininejad20a.pdf]: We don't require ordering which according to this is the main challenge. This might be a reason that this would work for our case?