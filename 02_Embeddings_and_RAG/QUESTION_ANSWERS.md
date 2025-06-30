# Session 2 Assignment 

#### ❓Question #1:

The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 

1. Is there any way to modify this dimension?

Yes, according to the documentation you can specify the dimension parameter to reduce the output size by passing in 
```
{
  "model": "text-embedding-3-small",
  "input": "Your text here",
  "dimensions": 512
}
```
for example:
```
response = openai.embeddings.create(
    model="text-embedding-3-small",
    input="Your text here",
    dimensions=512
)
```

2. What technique does OpenAI use to achieve this?

OpenAI uses a very common technique for dimensionality reduction called PCA - Principal Component Analysis. Josh Starmer has a very good video on PCA - https://youtu.be/FgakZw6K1QQ 
> NOTE: Check out this [API documentation](https://platform.openai.com/docs/api-reference/embeddings/create) for the answer to question #1, and [this documentation](https://platform.openai.com/docs/guides/embeddings/use-cases) for an answer to question #2!

#### ❓Question #2:

What are the benefits of using an `async` approach to collecting our embeddings?

##### Answer
Async is better for collecting embeddings because it allows you to process many requests in parallel. Why? We essentially send over a list of text to OpenAI to obtain the embeddings. If we made the call synchronous, it would process the list one item at a time which would be much slower. 

> NOTE: Determining the core difference between `async` and `sync` will be useful! If you get stuck - ask ChatGPT!

#### ❓ Question #3:

When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

##### Answer

There are the parameters `temperature` and `top_p` that are most widely used hyper-parameters to control the amount of randomness in the output. https://www.youtube.com/watch?v=xKoDAFm7B_c has a very good explanation. A `temperature` of 0 or a `top_p` value of 0 will make the outputs reproducible. According to https://www.youtube.com/watch?v=d7om0yPl5y8 we should not vary both `temperature` and `top_p` both at the same time. There is also the `seed value` that you can use for some models to make output reproducible. 

> NOTE: Check out [this section](https://platform.openai.com/docs/guides/text-generation/) of the OpenAI documentation for the answer!

#### ❓ Question #4:

What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

What is that strategy called?

##### Answer

Chain of thought prompting

> NOTE: You can look through ["Accessing GPT-3.5-turbo Like a Developer"](https://colab.research.google.com/drive/1mOzbgf4a2SP5qQj33ZxTz2a01-5eXqk2?usp=sharing) for an answer to this question if you get stuck!

