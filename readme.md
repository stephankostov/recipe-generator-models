# Development Process

Here the module was created without nbdev since most of the work was done in the .py files. Notebooks here were used for initial exploration and evaluation.

# Design

## Problem Scope

Our task at hand here is to generate recipes, which has two core requirements: to generate food tokens, and their respective quantities. For these two requirements we need to either design a model which is able to predict both, or split the task up and have two models for each requirement.

### Split Models

Ultimately this is the more simple approach, for two core reasons:

- Implementation: It's more straightforward to split the task up and train specific models for each.
- Maintainability: Models can be retrained individually when required.

### Singular Model

Ideally this would be what to strive for as it will be able to capture the dependancy between tokens and their quantities the generation of each.

However the implementation is difficult here. How can we encode this information of token_id and quantity into the model, to be able to encode and decode both. 

## Model Architectures 

Conveniently, what makes sense here is a Transformer architecture. The model needs to understand information of its context ie. the previous ingredients presented to it. This understanding includes an encoding of what that ingredient is, as well as the relationships between them. The encoding here is already done through our data (molecular composition), it's the relationships between each in the context that we require from the architecture, which is precisely what the attention mechanism is for. In particular this mechanism is able to capture relationships of many levels of complexity (multi-heads), and by its nature entirely independent of their positonal distance.

The idea behind this project was heavily inspired by LLM's and my desire to work with them, so it makes sense that this ends up being the same problem framing as them.

## Differences From LLM's

- Using vectorised molecular compound data as embeddings
- Removing the positional embeddings

## Types of Tansformers

The transformer architecture is an obvious choice, but from here which type of transformer would be best? There are two main decisions to make when condisidering this:

- encoder-decoder vs decoder-only
- autoregressive vs non-autoregressive

### Encoder-Decoder vs Decoder-only

The intuition behind this decision is whether or not we need an additional block in order to understand the input data seperate from that which is generating output data. 

Do we expect our input data to be different from our output data? Lets take for our simplest case of generating the ingredient tokens. Our input data is simply just a set of previous tokens, and the outputs are those with the addition of the newly generated tokens. 

### Autogressive vs Non-Autoregressive

It's important to note that this is referring to the generation mechanism at inference time. Do we want the model to generate tokens one at a time, or do we want the model to output the completed sequence all at once? 

Do we want the model to iteratively get to the final output, or do we want it to generate it all at once. Treating the model as an all-knowing black-box we would think that it should be able to generate the right answer at once. When viewing the model's generation as thinking, intuitively, a model should be more accurate if allowed to predict each token with consideration to all others in the sequence - those prior as well as the ones after. In other words, it should be able to change its predictions of tokens beforehand depending on those it chooses later.

But this assumption of the model thinking could very well be flawed. Rather than thinking it's moreso doing a one-step calculation to come up with everything at once. Maybe allowing it to iterate is more close to thinking as we know it, as thought isn't a one-step process of neurons, it's many over time which evolves and forms the final product. 

Intuition here can only go so far and gets complicated fast. Empirically it is known that for token generation, autoregressive geneartion gives more accurate results, at the expense of haivng more of an inference cost as inference is done for each token. 

Since this application is very constrained - to recipes with 1k ingredient tokens rather than text about anything known to the internet - the model can be small, and therefor inference will not be costly so should not be a concern. We are much more interested in accuracy, and therefor when knowing this fact autoregressive models are an easy choice.

When it comes to quantity prediction though, it makes sense to be able to predict every quantity at once, as the quantities are done in proportion to the whole recipe.  Could this be a flaw in the methodology here - is it flawed that the outputs of each token are dependent on eachother and the recipe as a whole? 

## Quantity Model

The quantity of the model is generated in an non-autoregressive method, which generates the weights of each ingredient in the recipe. 

This might be flawed when training though, as there may be many different weight combinations of a recipe. Take a cake for example, there are many different concentrations of flour/sugar/butter/water corresponding to different types of cakes. If this is the case the model would learn to generate mean values for each, resulting in the model predicting a simple heuristic of even proportions for each ingredient. 

### Proportion Dependancy

The way the data is set up, as proportions of the total recipe, we must generate the full result at once because to the loss can only be calculated once the full recipe has been generated. 

We could remove this dependancy, and get the model to generate weights instead, in an autoregressive manner. 

## Molecular Composition Encoding

Why can't we just use an LLM assistant to do this? Practically, we simply can and it will work great, but this project is really for practice building generative models so this was the niche. With this meaningful encoding there is also the potential for adding much greater interpretability of the model's decision making, to possibly provide insights on what makes a good recipe. 

Now how can we incorporate this molecular composition data into the model? As the numerical vectorised data that the model requires as input, we can simply feed in our tokens as vectors corresponding to the (scaled) concentration for each sample (ie. ingredient).

