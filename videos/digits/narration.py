"""Narration for the digits video. One string per scene; every number comes from results.json.
    python narration.py          -> audio/ + timing.json
    python narration.py --cues   -> list sentences with cue numbers"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # 0 a digit is 64 numbers
    "Here's a handwritten digit. "
    "You can see it's a zero. "
    "But a computer doesn't see a zero. It sees this. "
    "An eight by eight grid of brightness values, from zero for blank, to sixteen for full ink. "
    "Number the squares row by row, from zero to sixty three. "
    "That's the whole input. Sixty four numbers, and none of them means zero on its own. "
    "Models learn from thirteen hundred and forty seven digits like this, written by real people. "
    "Four hundred and fifty more are kept aside, to test on digits they've never seen.",
    # 1 a decision tree
    "First, a decision tree. It plays twenty questions, one pixel at a time. "
    "Its first question is, is pixel thirty six lit at all? "
    "That's this square, near the middle. "
    "It was picked because zeros have a hole right there. "
    "Then it asks about another pixel. Here is a tree three questions deep. "
    "Take this zero. Pixel thirty six is dark. Pixel twenty eight, dark. Pixel twenty one, lit. "
    "The tree says zero. Correct. "
    "Now this two. Pixel thirty six has a brightness of just three. Barely there, but that still counts as yes. "
    "Twenty one is lit. Thirty three is dark. So the tree says eight. Wrong. "
    "And that leaf is a mess. Of the six hundred and fifty six training digits that end up there, only nineteen percent are eights.",
    # 2 accuracy and why
    "On the test digits, this three question tree gets forty seven percent right. "
    "Let the tree grow as deep as it likes, and it ends up fourteen questions deep, with a hundred and forty seven leaves. "
    "That gets eighty five point six percent. "
    "Why so many? Because one pixel tells you very little. "
    "Here are two nines from the training data. "
    "In this one, pixel thirty six is dark. In this one, it's bright. "
    "Same digit, opposite answers to the very first question. "
    "Write a digit a little lower, or tilt it, and the ink lands on different squares. "
    "So the tree has to patch together lots of special cases, one pixel at a time.",
    # 3 hidden neurons
    "A neural network asks a different kind of question. "
    "Each hidden neuron looks at all sixty four pixels at once, with one weight per pixel. "
    "Sixty four weights fit in an eight by eight grid too. "
    "Yellow means ink on that square pushes the neuron up. Blue means ink there pushes it down. "
    "Multiply each pixel by its weight. "
    "Add it all up with a bias, and squash the total to between zero and one. "
    "For this zero, the answer is point nine four. "
    "Our network has sixteen of these hidden neurons. Here are four of them, after training. "
    "This one likes ink along the bottom. "
    "This one likes strokes down both sides, with nothing in the middle. "
    "This one likes a diagonal stroke. "
    "Not all sixteen are this easy to read, but each one answers a question about the whole image. Does it contain my pattern? "
    "And because it weighs every pixel together, no single square decides its answer.",
    # 4 bottom-stroke neuron on three digits
    "Let's test the bottom stroke neuron. "
    "The zero gets point nine four. "
    "This eight gets point nine eight. "
    "And this four, which has no stroke along the bottom, gets point zero six.",
    # 5 output layer + flow
    "Next comes the output layer. Ten neurons, one for each digit. "
    "Each one takes the sixteen hidden answers as its inputs, with its own weights. "
    "Take the bottom stroke answer. The output for two gives it a weight of plus one. "
    "The outputs for four and seven give it minus one point eight. Those digits don't have a bottom stroke. "
    "At the end, the ten scores are turned into chances that add up to one. "
    "Let's run the four through. "
    "Sixty four pixels go in. "
    "Sixteen hidden neurons answer. "
    "And the output for four gets ninety five percent. "
    "Now the two that fooled the small tree. "
    "The network says two, at sixty nine percent, with three as the runner up at twelve. Less sure, but right.",
    # 6 training
    "But nobody drew those patterns. The network started with random weights. Here are all sixteen neurons at the start. Pure noise. "
    "On the test digits, it scores ten percent, a blind guess. "
    "Training shows it the training digits, and nudges every weight a little, to make the right answer more likely. "
    "One pass through all of them is called an epoch. "
    "After one epoch, thirty percent. "
    "After five, eighty four. "
    "After twelve, ninety three point six. "
    "The noise fades, and strokes appear. "
    "It wobbles a bit from epoch to epoch. "
    "After sixty epochs, it scores ninety three point one percent on digits it has never seen.",
    # 7 scoreboard
    "So here's the scoreboard, on the same four hundred and fifty test digits. "
    "The full decision tree, eighty five point six percent. "
    "The network, ninety three point one. "
    "The tree misses sixty five digits. The network misses thirty one. "
    "Here are some the tree got wrong, and the network got right. "
    "A one, that the tree called a four. An eight, that it called a five. A four, that it called a nine. "
    "To be fair, it goes both ways. Thirteen digits fooled the network but not the tree. "
    "But forty seven went the other way.",
    # 8 close fair
    "So are networks just better? Not always. "
    "A tree's questions work well when each input already means something on its own. "
    "Think of a spreadsheet, with columns like hours studied, or age, or price. "
    "Is hours studied above some cutoff, is a useful question by itself. "
    "On data like that, trees, especially many trees combined, are often as good as a network, or better. "
    "A single pixel isn't like that. "
    "The meaning lives in combinations. A stroke, a loop, a curve. "
    "And that's exactly what hidden neurons are built to find. "
    "Which is why networks win on raw inputs like pixels, sound, and text.",
]


def check_numbers():
    """Fail loudly if results.json no longer matches what the narration says."""
    r = json.loads((HERE / "results.json").read_text())
    pct = lambda v: round(100 * v, 1)  # noqa: E731
    assert (r["n_train"], r["n_test"]) == (1347, 450)
    assert r["root_pixel"] == 36 and r["root_thr"] == 0.5
    assert round(100 * r["tree_depth3_acc"]) == 47
    assert (r["tree_full_depth"], r["tree_full_leaves"]) == (14, 147)
    assert pct(r["tree_full_acc"]) == 85.6 and pct(r["net_acc"]) == 93.1
    curve = {c["epoch"]: c["test"] for c in r["curve"]}
    assert [round(100 * curve[e]) for e in (0, 1, 5)] == [10, 30, 84] and pct(curve[12]) == 93.6
    leaf8 = r["tree"]["yes"]["yes"]["no"]
    assert leaf8["pred"] == 8 and leaf8["n"] == 656 and round(100 * leaf8["purity"]) == 19
    w0, w4, w8 = (next(w for w in r["walk"] if w["label"] == d) for d in (0, 4, 8))
    bad = r["small_tree_wrong"]
    assert bad["label"] == 2 and bad["tree_small_guess"] == 8 and bad["tree_small_path"][0]["value"] == 3
    assert [s["pixel"] for s in w0["tree_small_path"]] == [36, 28, 21]
    assert [s["pixel"] for s in w4["tree_small_path"]] == [36, 21, 33]
    assert [round(w["hidden"][1], 2) for w in (w0, w8, w4)] == [0.94, 0.98, 0.06]
    assert round(r["out_weights"][1][2], 1) == 1.0 and round(r["out_weights"][1][4], 1) == -1.8 \
        and round(r["out_weights"][1][7], 1) == -1.8
    assert round(100 * w4["net_probs"][4]) == 95
    assert bad["net_guess"] == 2 and round(100 * bad["net_probs"][2]) == 69 and round(100 * bad["net_probs"][3]) == 12
    assert round(r["root_yes_by_digit"]["9"], 2) == 0.68
    assert r["n_tree_misses_net_gets"] == 47 and r["n_net_misses_tree_gets"] == 13
    assert r["n_tree_misses_net_gets"] + r["n_both_miss"] == 65 and r["n_net_misses_tree_gets"] + r["n_both_miss"] == 31
    assert [(w["label"], w["tree_full_guess"]) for w in r["tree_misses_net_gets"][:2]] == [(1, 4), (8, 5)]
    assert (r["tree_misses_net_gets"][3]["label"], r["tree_misses_net_gets"][3]["tree_full_guess"]) == (4, 9)


if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        check_numbers()
        narrate(SEGMENTS, HERE, voice="am_michael", engine="kokoro", speed=1.1)
