"""Large language models from first principles, covering all three pages of Google's Machine Learning Crash Course
module "Introduction to Large Language Models" (CC BY 4.0). One string per scene; sentences become cue points.

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
sys.path.insert(0, str(HERE))
from vidkit import narrate, print_cues  # noqa: E402
from toy import BLEND, ORANGE_NEXT, ORANGE_TOTAL, SCORES, SHRINK, WEIGHTS  # noqa: E402

SEGMENTS = [
    # ===== 0 hook + roadmap
    "When you type a question into a chatbot, and it writes back a whole essay, what is actually going on inside? "
    "The surprising answer is that it is playing one very simple game, over and over. "
    "Guess the next word. "
    "In this video we'll build up to large language models from scratch, following Google's Machine Learning Crash Course. "
    "We'll start with the game itself. "
    "Then the older ways of playing it, and why each one fell short. "
    "Then the Transformer, the design behind today's models. "
    "And finally, how those models get trained, tuned, prompted and used, and where they go wrong.",

    # ===== chapter 1: the game
    # 1 language model + probability
    "Start with the phone keyboard in your pocket. "
    "Type I'll be home, and it suggests soon, late, or by. "
    "To do that, it needs a sense of how likely each word is to come next. "
    "That is exactly what a language model is. "
    "A language model estimates the probability of a piece of text appearing within a longer sequence. "
    "A probability is just a number between zero and one hundred percent, saying how often you'd expect something, out of all the possibilities.",
    # 2 tokens
    "Computers don't read words the way we do, so first the text is chopped into pieces called tokens. "
    "A token can be a whole word, like kitchen. "
    "It can be a piece of a word, called a subword, like un, believ, and able. "
    "Or it can be a single character. "
    "Subwords are a handy middle ground. "
    "A long or rare word can be built from common pieces, so the model never meets a word it can't spell out. "
    "To keep things simple, we'll mostly treat each word as one token.",
    # 3 the rain table
    "Here is the course's example. "
    "When I hear rain on my roof, I, blank, in my kitchen. "
    "A language model looks at the words around the blank, and gives every possible filler a probability. "
    "Cook soup gets nine point four percent. "
    "Warm up a kettle, five point two. "
    "Cower, three point six. Nap, two point five. And relax, two point two. "
    "Notice the filler can be more than one token. "
    "It could be a phrase, a sentence, or even a whole essay. "
    "And these don't add up to a hundred, because thousands of other fillers share the rest.",
    # 4 picking + tasks
    "A table of probabilities isn't an answer yet. "
    "An application still has to pick one. "
    "It can simply take the most likely option, cook soup. "
    "Or it can choose at random among the options above some threshold, which gives more varied text. "
    "That one skill, filling in a blank, stretches surprisingly far. "
    "Pick a word, add it to the text, and predict again, and you are generating text. "
    "Make the blank the French version of an English sentence, and you are translating. "
    "Make it a short version of a long document, and you are summarizing. "
    "By learning the statistical patterns of tokens, a model ends up building a rich internal picture of language, good enough to produce text that sounds right.",

    # ===== chapter 2: counting
    # 5 n-grams defined
    "So where do the probabilities come from? "
    "The oldest answer is to count. "
    "An N-gram is a run of N words in a row, taken from training text. "
    "Take the phrase, you are very nice. "
    "Slide a window two words wide across it, and you get three 2-grams, also called bigrams. "
    "You are. Are very. Very nice. "
    "Make the window three words wide, and you get two 3-grams, or trigrams. "
    "You are very. And are very nice.",
    # 6 counting orange is
    "Now, to predict a word, a trigram model looks at the two words before the blank. "
    "Say those words are orange is. "
    "The model finds every trigram in its training text that starts with orange is, and counts what came third. "
    f"In a small made-up example, ripe came next {ORANGE_NEXT[0][1]} times, cheerful {ORANGE_NEXT[1][1]}, "
    f"sweet {ORANGE_NEXT[2][1]}, and bright {ORANGE_NEXT[3][1]}. "
    f"That's {ORANGE_TOTAL} in total. "
    f"Divide each count by {ORANGE_TOTAL}, and the counts become probabilities. "
    f"Ripe gets {round(100 * ORANGE_NEXT[0][1] / ORANGE_TOTAL)} percent, and cheerful gets {round(100 * ORANGE_NEXT[1][1] / ORANGE_TOTAL)} percent. "
    "That's a working language model, built from nothing but counting.",
    # 7 context
    "But there's a problem hiding in that example. "
    "Orange is ripe is about the fruit. "
    "Orange is cheerful is about the colour. "
    "Which one is right depends on what came earlier. "
    "That earlier information is called context. "
    "People are very good with context. "
    "Watching the third act of a play, you still remember the characters from act one. "
    "A joke's punchline only works because you remember the setup. "
    "For a language model, context is any helpful information before or after the token it is predicting. "
    "If the text began with, I peeled the orange, the answer would be obvious. "
    "But a trigram only ever sees two words of context, so models built on trigrams make a lot of mistakes.",
    # 8 sparsity
    "So why not just use longer N-grams, and see more context? "
    "Because the longer a sequence of words is, the rarer it is. "
    "Orange is shows up all the time. "
    "The orange is, less often. "
    "She said the orange is, almost never. "
    "As N grows, each exact sequence appears fewer and fewer times in the training text. "
    "When N gets very large, nearly every sequence appears just once. "
    "A count of one can't tell you anything about what usually comes next. "
    "Counting has hit a wall. We need something that can learn context, not just count it.",

    # ===== chapter 3: neural networks
    # 9 neural network refresher + embeddings
    "That something is a neural network. "
    "A quick reminder of what that is. "
    "A neural network is a big calculation full of adjustable numbers, called parameters, or weights. "
    "Numbers go in, get multiplied by the weights, added up and passed along, and a prediction comes out. "
    "Training means showing it examples, measuring how wrong it is, and nudging every weight a little to make it less wrong. "
    "The method that works out those nudges is called backpropagation. "
    "But a network only understands numbers, so each token is first turned into a list of numbers, called an embedding. "
    "Tokens with similar meanings get similar lists.",
    # 10 RNN
    "The first networks built for language were recurrent neural networks. "
    "A recurrent network trains on a sequence of tokens, reading them one at a time. "
    "It keeps a running memory. "
    "At each step, it mixes the new token into that memory, and passes the memory on to the next step. "
    "Over training, it learns what to hold on to, and what to ignore. "
    "That's a lot like how you listen to someone speak. "
    "A large recurrent network can carry context across a passage of several sentences, far more than any N-gram.",
    # 11 RNN limits + vanishing gradient
    "But it still has limits. "
    "Everything it knows about the past has to squeeze through that one memory, token by token. "
    "By the end of a long passage, the early words have mostly faded. "
    "Training has a matching problem, called the vanishing gradient problem. "
    "The signal that tells the network how to adjust its weights has to travel backwards through every step. "
    "At each step it gets multiplied by a number, often smaller than one. "
    f"Multiply by one half, ten times over, and you are left with about one thousandth of the signal. "
    "So the network barely learns anything from words far back in the text. "
    "What we really want is a model that can look at the whole context at once.",

    # ===== chapter 4: transformers
    # 12 LLM + transformer
    "That's exactly what large language models do. "
    "A large language model, or LLM, predicts a token or a sequence of tokens, sometimes many paragraphs long. "
    "It beats N-grams and recurrent networks for two reasons. "
    "It has far more parameters. "
    "And it gathers far more context, evaluating the whole context at once instead of token by token. "
    "The most successful design for building LLMs is called the Transformer.",
    # 13 encoder decoder
    "Here's a Transformer doing translation, from English to French. "
    "A full Transformer has two halves. "
    "The encoder takes the input text, and turns it into an intermediate representation, a set of numbers that captures what the text means. "
    "The decoder takes that representation, and turns it into useful output text, here the French sentence. "
    "Each half is an enormous neural network in its own right. "
    "Many popular LLMs keep only the decoder half, since all they need to do is keep writing.",
    # 14 self-attention question + self
    "The trick that gives Transformers so much context is called self-attention. "
    "For every token in the input, self-attention asks one question. "
    "How much does each other token affect the way I should interpret this one? "
    "The self means the tokens only look at their own input sequence. "
    "Other kinds of attention compare tokens to a different sequence, like the words of a translation being written. "
    "Self-attention only weighs relationships within the input itself.",
    # 15 animal street
    "Here's the course's example. "
    "The animal didn't cross the street because it was too tired. "
    "That's eleven words, and each one pays attention to the other ten, asking how much each matters to it. "
    "Look at the word it. "
    "Pronouns are ambiguous. Does it mean the animal, or the street? "
    "Self-attention scores how relevant each word is to it, and here animal scores higher than street. "
    "Now change the last word. Because it was too wide. "
    "A street can be wide, an animal usually isn't, so this time attention should favour street. "
    "One word, at the far end of the sentence, flips the meaning, and attention picks it up.",
    # 16 how attention is computed
    "How does it actually compute those scores? "
    "Remember that every word is a list of numbers. "
    "Let's pretend each list has just two numbers, so we can draw them as arrows. "
    "To score how well two words match, multiply their numbers pair by pair, and add. "
    "That's called a dot product. Arrows pointing the same way score high. "
    f"It and animal score {SCORES[0]:.0f}. It and tired score {SCORES[2]:.0f}. It and street score just {SCORES[1]:.1f}. "
    "Next, turn the scores into weights that add up to one, using a function called softmax. "
    f"Animal gets {WEIGHTS[0]:.2f}, tired {WEIGHTS[2]:.2f}, and street {WEIGHTS[1]:.2f}. "
    "Finally, build a new arrow for it, by blending the other arrows using those weights. "
    "The new it points mostly towards animal. It now carries the meaning of what it refers to. "
    "Real models first pass each list through learned weights, called queries and keys, but the idea is exactly this.",
    # 17 directions
    "Attention can also run in one direction, or both. "
    "Bidirectional attention lets a word look at the words before it and after it, like it looking ahead to tired. "
    "Unidirectional attention lets a word look only at the words on one side, the ones before it. "
    "Why limit it? Because when you generate text one token at a time, the later words don't exist yet. "
    "So encoders, which build a picture of a whole sequence, use bidirectional attention. "
    "And decoders, which generate token by token, use unidirectional attention.",
    # 18 multi-head multi-layer
    "One attention calculation can only capture one kind of relationship. So Transformers run several at once. "
    "Each self-attention layer is made of multiple attention heads. "
    "Every head starts with its own random parameters, so during training each one drifts toward a different job. "
    "One head might learn which noun a pronoun refers to. "
    "Another might learn the grammar between words. "
    "The layer then combines all the heads' outputs, with a calculation like a weighted average. "
    "Then layers are stacked. The output of one layer becomes the input of the next. "
    "Early layers tend to pick up basic syntax. "
    "Deeper layers build on that, to grasp sentiment, context, and themes running across the whole text.",
    # 19 why so large
    "All of this adds up to a lot of parameters. "
    "Transformers can have hundreds of billions, or even trillions of them. "
    "That goes against earlier advice in the course, which favours smaller models, since they need fewer resources to make predictions. "
    "But for Transformers, research keeps finding the same thing. "
    "Models with more parameters consistently outperform models with fewer.",

    # ===== chapter 5: generating
    # 20 training objective
    "So how does an LLM learn? "
    "By playing the guessing game, billions of times. "
    "Take a sentence from the training text, hide a word, and ask the model to predict it. "
    "Compare its guess to the real word, and nudge the parameters with backpropagation. "
    "The neat part is that nobody has to label anything. "
    "The text supplies its own answers, which is why this counts as unsupervised learning.",
    # 21 autocomplete
    "Predicting a missing word or two might sound unimpressive. It's basically autocomplete. "
    "And that's the point. An LLM is an autocomplete that can predict thousands of tokens. "
    "Take the sentence, my dog, Max, knows how to perform many traditional dog tricks, followed by a hidden sentence. "
    "The model gives probabilities to whole sentences. "
    "For example, he can sit, stay, and roll over, gets three point one percent. "
    "For example, he knows how to sit, stay, and roll over, gets two point nine.",
    # 22 questions as masks + math
    "A chat works the same way. "
    "Your question is the given text, followed by an imaginary blank. "
    "What is the easiest trick to teach a dog? "
    "The model generates probabilities for possible responses, picks a token, adds it, and goes again, until the answer is done. "
    "The same is true of models trained on huge numbers of math word problems. "
    "They can look like they're reasoning, but underneath, they are autocompleting the problem.",
    # 23 benefits
    "That simple game turns out to be very useful. "
    "LLMs can write clear, easy to understand text for almost any audience. "
    "They make good predictions on the tasks they were explicitly trained on. "
    "Some researchers claim LLMs can also handle tasks they were never trained on, but other researchers dispute that claim.",
    # 24 training problems
    "But LLMs come with real problems, starting with training. "
    "You need an enormous training set. "
    "Training takes months, and huge amounts of computing power and electricity. "
    "And no single machine can do it alone. "
    "The work is split across thousands of chips, like Google's Tensor Processing Units, and keeping them all working together, in parallel, is a hard engineering problem of its own.",
    # 25 inference problems
    "Using a trained model, which is called inference, has problems too. "
    "LLMs hallucinate. Their predictions often contain mistakes, stated just as fluently as the facts, because the model produces likely text, not checked text. "
    "Every prediction also costs compute and electricity. "
    "There's a trade-off here. Training on more data usually makes a model cheaper to run, but makes training itself more expensive. "
    "And like any machine learning model, an LLM can pick up all sorts of bias.",

    # ===== chapter 6: making it useful
    # 26 foundation models
    "A model trained on general text like this goes by several names. "
    "A foundation model, a base model, or a pre-trained model. "
    "It has read enough language to know a remarkable amount about grammar, words and idioms. "
    "It can write helpful sentences about the topics it has seen, and even creative work, like poetry. "
    "But many everyday machine learning problems aren't about writing. "
    "Regression, predicting a number. Classification, putting things into categories. "
    "For those, a foundation model is a platform to build on, not a finished solution.",
    # 27 fine-tuning
    "Turning it into a solution takes fine-tuning. "
    "Foundation models are so good at spotting patterns that they need surprisingly little extra training to learn a specific task. "
    "Fine-tuning trains the model further, on examples of exactly the task your application will do. "
    "Sometimes just a few hundred, or a few thousand examples are enough. "
    "For instance, emails labelled urgent or not urgent. "
    "The fine-tuned model makes better predictions on that task than the foundation model it came from.",
    # 28 PET + size
    "Even with so few examples, standard fine-tuning is expensive. "
    "On every backpropagation step, it updates every single parameter in the model. "
    "A smarter approach, called parameter-efficient tuning, adjusts only a small subset of the parameters on each step. "
    "Either way, notice what doesn't change. The size. "
    "If the foundation model has ten billion parameters, the fine-tuned model has ten billion parameters too.",
    # 29 distillation
    "That's a lot of model, and much of it is wasted on any one application. "
    "Large swaths of those parameters are irrelevant to a specific task, but they still cost compute every time the model runs. "
    "Distillation creates a smaller version. "
    "A small model, with far fewer parameters, is trained to reproduce the big model's predictions. "
    "The distilled model makes predictions much faster, and needs far fewer computational and environmental resources. "
    "The catch is that its predictions usually aren't quite as good, because more parameters almost always means better predictions.",
    # 30 prompt engineering
    "Fine-tuning and distillation are done by engineers. "
    "But the people using a model can steer it too, just by how they write the prompt. "
    "That's called prompt engineering. "
    "Humans learn well from examples, and so do LLMs. "
    "Say you want the model to name a fruit's botanical class. "
    "Ask it, apple, with no examples, and that's zero-shot prompting. "
    "Without context, it might tell you about the technology company. "
    "Show it one example first, peach is a drupe, and that's one-shot prompting. Now it answers, apple, pome. "
    "Sometimes one example isn't enough, so you show several. Plum, drupe. Pear, pome. Then lemon. "
    "That's few-shot prompting, and the examples act like a tiny training set. "
    "Crucially, prompting doesn't change a single parameter. It only uses the patterns the model already knows.",
    # 31 offline inference
    "One last practical trick. "
    "Normally a model answers each request the moment it arrives. That's online inference. "
    "But some LLMs are so large that answering live is too slow for jobs like classification. "
    "So teams use offline inference instead, also called bulk or static inference. "
    "The model makes its predictions in advance, and the results are stored, ready to look up. "
    "If the job only needs doing once a week, it doesn't matter how long it takes. "
    "For example, Google Search used an LLM offline to build a list of over eight hundred synonyms for Covid vaccines, in more than fifty languages. "
    "Live searches were then matched against that cached list.",
    # 32 responsibility
    "Finally, a model learns from its data, including the bad parts. "
    "An LLM shares the biases of the data it was trained on, and of the data it was distilled on. "
    "So use LLMs fairly and responsibly, following the course's guidance on data and fairness.",
    # 33 recap
    "Let's put it all together. "
    "A language model assigns probabilities to the next token. "
    "N-grams did that by counting, but couldn't see enough context. "
    "Recurrent networks carried context forward, but it faded. "
    "Transformers use self-attention to weigh every token against every other, all at once, across many heads and layers, with billions of parameters. "
    "Trained to fill in blanks, they become autocomplete at a massive scale. "
    "Fine-tuning adapts them to a task, distillation makes them smaller, and prompting steers them without changing a thing. "
    "And through it all, remember they can hallucinate, they are costly, and they can be biased. "
    "You'll find the full module, with exercises, in Google's Machine Learning Crash Course.",
]


def check_numbers():
    assert ORANGE_TOTAL == 12 and [round(100 * c / 12) for _, c in ORANGE_NEXT[:2]] == [50, 25]
    assert [round(s, 1) for s in SCORES] == [2.0, 0.5, 1.0]
    assert [round(w, 2) for w in WEIGHTS] == [0.63, 0.14, 0.23]
    assert BLEND[0] > BLEND[1] > 0
    assert round(SHRINK[10], 3) == 0.001


if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        check_numbers()
        narrate(SEGMENTS, HERE, voice="am_michael", engine="kokoro", speed=1.1)
