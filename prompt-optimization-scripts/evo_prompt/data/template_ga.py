templates_2 = {
    "hallucination": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: You will analyze a query, reference text, and answer to determine if the answer contains hallucinations. Label as "factual" if supported by reference text, or "hallucinated" if it contains unsupported claims.
Prompt 2: Examine if an answer accurately reflects its reference text without adding unsupported information. Return JSON with label and explanation.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Analyze a query, reference text, and answer to determine if the answer contains only information from the reference text. Label as "factual" if fully supported, or "hallucinated" if it contains unsupported claims.
2. <prompt>Evaluate if an answer to a query contains only information supported by the reference text. Label as "factual" if all claims are supported, or "hallucinated" if any information is unsupported or contradictory. Return a JSON response with "label" and "explanation" fields.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
    "cls": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Rewrite the complex text into simpler text while keeping its meaning.
2. <prompt>Transform the provided text into simpler language, maintaining its essence.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
    "sim": """Please follow the instruction step-by-step to generate a better prompt.  
1. Crossover the following prompts to generate a new prompt:  
Prompt 1: Your task is to classify the comment as one of the following categories: terrible, bad, okay, good, great.
Prompt 2: In this task, you are given sentences from movie reviews. The task is to classify a sentence as one of the following categories: terrible, bad, okay, good, great.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: In this task, you are given comments from movie reviews. Your task is to classify each comment as one of the following categories: terrible, bad, okay, good, great.
2. <prompt>Given a sentence from a movie review, classify it into one of the following categories: terrible, bad, okay, good, or great.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
    "sum": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
    "qa": """Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: Rewrite the input text into simpler text.
Prompt 2: Rewrite my complex sentence in simpler terms, but keep the meaning.
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. Crossover Prompt: Simplify the complex text while maintaining its meaning.
2. <prompt>Simplify the complex text while maintaining its meaning.</prompt>

Please follow the instruction step-by-step to generate a better prompt.
1. Crossover the following prompts and generate a new prompt:
Prompt 1: <prompt1>
Prompt 2: <prompt2>
2. Mutate the prompt generated in Step 1 and generate a final prompt bracketed with <prompt> and </prompt>.

1. """,
}
