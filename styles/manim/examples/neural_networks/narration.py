"""Narration for the neural-networks example. One string per scene.
    python narration.py          -> audio/ + timing.json
    python narration.py --cues   -> list sentences with cue numbers"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # 0 the problem
    "Say you want to predict whether a student will pass an exam. "
    "You know two things about each one. How many hours they studied, and how many hours they slept. "
    "Here are ten students, and whether they passed. "
    "Let's put each one on a chart. "
    "Hours studied go across. Hours slept go up. "
    "Green passed. Red failed. "
    "What we want is a rule that can tell green from red.",
    # 1 a weighted score
    "The simplest rule you could write is a score, like a formula in a spreadsheet. "
    "Take hours studied, times a weight. "
    "Add hours slept, times another weight. "
    "Then subtract a bar that the score has to clear. "
    "The weights say how much each thing matters. "
    "The bar is called the bias. "
    "If the score comes out above zero, predict pass. "
    "Here, one times study, plus one times sleep, minus ten and a half. "
    "On the chart, that rule is a straight line. "
    "Everyone above it gets pass. Everyone below it gets fail.",
    # 2 squash -> neuron
    "One more tweak. "
    "Pass or fail is all or nothing, so you can't tell when a rule is getting closer to right. "
    "So instead, we squash the score into a number between zero and one. "
    "A big positive score becomes almost one. A big negative score becomes almost zero. "
    "Near the line, it's about a half. "
    "Now the answer is a chance of passing. "
    "And that's a neuron. "
    "Numbers come in. Each one is multiplied by a weight. "
    "They're added up with the bias. And the total is squashed.",
    # 3 one line is not enough
    "But look at Dev. "
    "He studied nine hours, slept three, and failed. "
    "Our line says he passed. "
    "And Jo slept nine hours, studied three, and failed too. "
    "Try to fix it. Tilt the line. Shift it. "
    "Whatever you do, at least two students land on the wrong side. "
    "One neuron can only draw one straight line. "
    "It can only ask one question. "
    "And this pattern needs two.",
    # 4 sub-questions
    "So break the problem into smaller questions. "
    "First question. Did they study enough? "
    "That's one neuron, with all of its weight on study, and none on sleep. "
    "Its line stands straight up, at about four and a half hours. "
    "Everyone to the right gets a yes. "
    "Second question. Did they sleep enough? "
    "Another neuron, with all its weight on sleep. "
    "Its line lies flat. Everyone above it gets a yes. "
    "Dev studied enough, but didn't sleep enough. "
    "Jo slept enough, but didn't study enough.",
    # 5 combine
    "Now add a third neuron, and give it the two answers as its inputs. "
    "Its question is, were both answers yes? "
    "It takes ten times each answer, and subtracts fifteen. "
    "Only two yeses clear that bar. One yes isn't enough. "
    "On the chart, that leaves just this corner. "
    "And now every student lands on the right side.",
    # 6 name the layers
    "What we just built is a neural network. And each column is a layer. "
    "The first column is the input layer. "
    "It's just the raw data, hours studied and hours slept. "
    "It doesn't calculate anything. It only holds the numbers going in. "
    "The last column is the output layer. It's the final answer. "
    "And the column in the middle is the hidden layer. That's where the smaller questions live. "
    "It's called hidden because it never appears in the data. "
    "Nobody gave us a column called studied enough. "
    "The network invents it in the middle, on the way to the answer. "
    "Bigger networks do the same thing, just more of it. "
    "Thousands of inputs, like the pixels in a photo. "
    "And many hidden layers, each one asking questions about the answers of the layer before.",
    # 7 random start + error
    "But I cheated. "
    "I picked those weights by hand, because I could see the answer on the chart. "
    "With real problems you can't, so the network has to find its own weights. "
    "It starts with random ones. "
    "The lines land anywhere, and the guesses are close to useless. "
    "To measure how wrong it is, take each guess, subtract the real answer, and square it. "
    "Add that up across all ten students. That's the error. "
    "One number. Lower is better.",
    # 8 slopes + backprop
    "Now, for each weight, ask one question. "
    "If I nudge this weight up a tiny bit, does the error go up, or down? "
    "That's the slope. "
    "Then move every weight a small step in whichever direction lowers the error. "
    "Working out all of those slopes at once, starting at the output and going back toward the inputs, is called backpropagation. "
    "One step barely changes anything. So we take twenty thousand of them.",
    # 9 training
    "Watch the lines. "
    "As they move, the error drops. "
    "By a thousand steps, the corner is taking shape. "
    "By twenty thousand, every student is on the right side. "
    "And look at what the hidden layer settled on. "
    "One neuron leans mostly on sleep. "
    "The other leans mostly on study. "
    "Nobody told it to ask those questions. "
    "It found its own versions of them, because that's what lowered the error.",
    # 10 new student + recap
    "Now ask about someone who was never in the data. "
    "Seven hours of study, four hours of sleep. "
    "The study question says yes. "
    "The sleep question says probably not. "
    "So the network's answer is point three. Probably fails. "
    "So that's the whole idea. "
    "A neuron is a weighted score, squashed between zero and one. "
    "Hidden layers let the network split a hard question into easier ones. "
    "And training means measuring the error, and nudging every weight downhill, over and over.",
]


def check_numbers():
    r = json.loads((HERE / "results.json").read_text())
    assert r["best_line_errors"] == 2
    assert round(r["new_out"], 1) == 0.3
    h = r["new_hidden"]
    assert h[1] > 0.9 and h[0] < 0.5  # study-neuron yes, sleep-neuron not sure
    assert all((o > 0.5) == bool(d[2]) for o, d in zip(r["final_out"], r["data"]))


if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        check_numbers()
        narrate(SEGMENTS, HERE, voice="en_US-lessac-high")
