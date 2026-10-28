"""Narration: Neural networks from first principles (all of Google MLCC's Neural networks module).
One string per scene; sentences become cue points [0], [1], ...

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # 0 the goal
    "Imagine you're building a smart thermostat. "
    "From two readings, temperature and humidity, you want it to tell whether a room feels comfortable. "
    "You could try to write the rules yourself. "
    "Or you could collect lots of examples, rooms where you measured both readings and asked people how they felt, "
    "and let the computer work out the rule. "
    "That's machine learning. "
    "Each reading we feed in is called a feature. "
    "The answer we want to predict, comfortable or not, is called the label. "
    "And the rule the computer learns is called a model. "
    "Step by step, we'll build up to a powerful kind of model: the neural network.",
    # 1 the chart
    "First, let's put the examples on a chart. "
    "Temperature goes across. Humidity goes up. "
    "Each dot is one room. "
    "Blue means people were comfortable. Orange means they weren't. "
    "Notice the shape. "
    "The comfortable rooms sit in the middle. "
    "Too hot, too cold, too damp, or too dry, and people complain. "
    "A model's job is to draw a boundary, with blue on one side and orange on the other.",
    # 2 the simplest rule
    "What's the simplest rule a computer could learn? "
    "Take each reading, and multiply it by a number that says how much that reading matters. "
    "Add the results together. "
    "Then add one more number, a starting offset. "
    "If the total is above zero, say comfortable. Otherwise, say not. "
    "The numbers that multiply the readings are called weights. "
    "The extra offset is called the bias. "
    "For this rule, learning just means finding good values for the weights and the bias.",
    # 3 it is a straight line
    "Here's the catch. "
    "The answer flips wherever the total is exactly zero. "
    "And for a rule built only from multiplying and adding, those points always form a straight line. "
    "Change the weights, and the line tilts. "
    "Change the bias, and it slides. "
    "But it is always straight. "
    "That's why this is called a linear model.",
    # 4 nonlinear
    "Now try to fence in the comfortable rooms with one straight line. "
    "Tilt it. Slide it. "
    "It never works, because the blue dots are surrounded on every side. "
    "When no straight line can separate the classes, the problem is called nonlinear. "
    "And most interesting real-world problems are nonlinear.",
    # 5 feature cross
    "One old trick is to invent new features. "
    "Take how far the temperature is from the middle, and multiply it by itself. "
    "Do the same for humidity. "
    "A linear model on those two new features can draw a circle, and the comfortable rooms are fenced in. "
    "Making a new feature by multiplying features together like this is called a feature cross.",
    # 6 the burden of guessing
    "But someone had to guess which features to make. "
    "Here's a different dataset. Blue in two opposite corners, orange in the other two. "
    "Squaring doesn't help here. "
    "This one needs the two features multiplied by each other. "
    "Every new problem means more guessing and testing. "
    "What we really want is a model that discovers the useful combinations on its own. "
    "That's exactly what a neural network does.",
    # 7 the linear model as a diagram, worked numbers
    "To build one, let's first redraw our linear model as a picture. "
    "Each input gets a circle. These circles are called nodes. "
    "Each line carries a weight. "
    "The output node multiplies each input by its weight, adds them up, and adds the bias. "
    "Let's try real numbers. "
    "Inputs of two and one. Weights of three and minus one. A bias of one half. "
    "Three times two, plus minus one times one, plus a half. "
    "The output is five and a half.",
    # 8 hidden layer + parameter count
    "Now the key idea. "
    "Instead of going straight from inputs to output, put a row of nodes in between. "
    "Each of these middle nodes is its own little linear model, with its own weights and its own bias. "
    "Each one computes its own weighted sum of the inputs. "
    "Then the output node takes a weighted sum of their results. "
    "The middle row is called a hidden layer, because we never see its values directly. "
    "We only see the inputs going in, and the answer coming out. "
    "Its nodes are often called neurons. "
    "All the weights and biases together are called the model's parameters. "
    "In the course's example, there are three inputs, four hidden neurons, and one output. "
    "Each hidden neuron has three weights plus a bias. That's four parameters, times four neurons, sixteen. "
    "The output has four weights plus a bias. Five more. "
    "Twenty one parameters in total, compared to just four for the linear model.",
    # 9 collapse
    "So, can this network draw curves? "
    "Surprisingly, no. "
    "Every step is still just multiplying and adding. "
    "Take a tiny example. A hidden node doubles its input. The output node triples that. "
    "The whole network simply multiplies by six. "
    "However many layers you stack, multiplying and adding on top of multiplying and adding is still just multiplying and adding. "
    "It collapses back into one linear model, and draws one straight line.",
    # 10 activation functions
    "The fix is to add a bend. "
    "After each neuron computes its weighted sum, pass the result through a simple nonlinear function, "
    "before handing it to the next layer. "
    "This is called an activation function. "
    "You may have met one already. "
    "Logistic regression uses the sigmoid function, which squashes any number into a value between zero and one. "
    "Tanh is similar, but squashes to between minus one and one. "
    "And the most popular is ReLU, short for rectified linear unit. "
    "Its rule is simple. If the number is negative, output zero. Otherwise, pass it through unchanged. "
    "ReLU usually works a little better than the smooth curves. "
    "It's cheaper to compute, and it helps avoid a training problem called vanishing gradients, which we'll meet soon.",
    # 11 why bends make shapes (real neurons)
    "Why does such a small bend help so much? "
    "Here's a real network, trained on our comfort data. "
    "It has one hidden layer of three ReLU neurons. "
    "Let's look at what each neuron responds to. "
    "Each one draws a straight line across the chart. "
    "On one side, it outputs zero. On the other side, its output grows the further you go. "
    "Neuron one. Neuron two. Neuron three. "
    "The output node adds these three together, each with its own weight. "
    "Combined, the three folds make a closed shape, a triangle wrapped around the comfortable rooms. "
    "Nobody hand-made a feature. The network found its own. "
    "With more neurons and more layers, each layer builds more complex features out of the layer before it, "
    "and the shapes can get as detailed as the problem needs.",
    # 12 ingredients + caveat
    "So a neural network has three ingredients. "
    "Nodes, arranged in layers. "
    "Weights and biases, connecting each layer to the next. "
    "And an activation function that bends each node's output. "
    "One honest caveat. "
    "Neural networks aren't always better than hand-made feature crosses. "
    "But they're a flexible option that works well on a huge range of problems.",
    # 13 inference walk-through
    "Using a trained network to make a prediction is called inference. "
    "Let's walk through one. "
    "First, we rescale the readings, so that zero is the middle of the range. "
    "Twenty six degrees becomes one point zero, and fifty five percent humidity becomes zero point five. "
    "Each hidden neuron computes its weighted sum. "
    "ReLU turns the negative one into zero, and passes the others through. "
    "The output node combines them, and a sigmoid turns the result into a probability. "
    "Zero point nine nine. "
    "A ninety nine percent chance this room is comfortable. "
    "Information only flows one way, forward, from the inputs to the output.",
    # 14 lessons from the interactive exercises
    "The course's exercises draw out three lessons from this. "
    "First, if every input were zero, every weight would be multiplying zero. "
    "Each hidden neuron would just output its bias, after ReLU. "
    "Second, because information flows forward, changing one layer can only affect the layers after it. "
    "Third, look at neuron one. Its sum was minus one point four seven, and it output zero. "
    "Had its sum been minus five, it would still output zero. "
    "A value can look the same even when the calculation behind it has changed, so check the calculations, not just the outputs.",
    # 15 loss
    "But where did those weights come from? "
    "Nobody chose them. They were learned. "
    "Training starts with random weights, so the first guesses are bad. "
    "To improve, the network needs a way to measure how bad. "
    "That measure is called the loss. "
    "For each example, compare the network's guess to the true label. "
    "A confident wrong answer gives a big loss. "
    "A confident right answer gives a loss near zero. "
    "Average over all the examples, and you get one number to make smaller.",
    # 16 gradient descent
    "Now picture every weight as a dial. "
    "For each dial, ask: if I turn it up a tiny bit, does the loss go up or down, and by how much? "
    "That answer is called the gradient. "
    "Then turn every dial a small step in the direction that lowers the loss. "
    "The size of that step is called the learning rate. "
    "Repeat, over and over. "
    "This is gradient descent.",
    # 17 backpropagation
    "The hard part is finding the gradient for every weight, when there can be millions of them, buried inside layers. "
    "That's what backpropagation does. "
    "It starts at the output, where the error is known. "
    "Then it works backwards, one layer at a time. "
    "Each layer's share of the blame is calculated from the layer after it. "
    "One backward pass gives the gradient for every weight at once. "
    "It's the most common way to train neural networks, and libraries like Keras do it for you automatically.",
    # 18 training montage (real)
    "Here's our network learning. "
    "At the start, with random weights, it gets fewer than half the rooms right. "
    "After sixty steps, the loss has dropped, and a shape is starting to form. "
    "After two hundred, every training room is classified correctly. "
    "By fifteen hundred, the boundary is crisp. "
    "We also kept forty rooms aside that the network never trained on. "
    "The loss on those is called the test loss. "
    "It's low too, so the network learned the pattern, not just memorized the examples. "
    "The course's own exercise solves a pattern like this with the same setup, one hidden layer of three ReLU neurons.",
    # 19 vanishing / exploding gradients
    "Training doesn't always go smoothly. "
    "Backpropagation multiplies many numbers together as it moves back through the layers. "
    "If those numbers are small, the product shrinks fast. "
    "The sigmoid's slope is never more than a quarter. "
    "Ten layers of that, and the signal is less than one millionth of its size. "
    "The early layers barely learn. "
    "This is the vanishing gradient problem. "
    "ReLU helps, because its slope is exactly one for every positive input. "
    "The opposite can happen too. "
    "If the weights are large, the product grows out of control, and training never settles. "
    "That's an exploding gradient. "
    "Batch normalization, or a lower learning rate, can help.",
    # 20 dead ReLU + dropout
    "ReLU has its own trap. "
    "If a neuron's sum gets pushed below zero for every example, it always outputs zero, and no gradient can flow through it. "
    "It can stay stuck forever. "
    "That's called a dead ReLU. "
    "A lower learning rate, or a variant like Leaky ReLU, which lets a little signal through below zero, can prevent it. "
    "Finally, a technique called dropout. "
    "During each training step, randomly switch off some of the neurons. "
    "That stops the network from leaning too heavily on any one of them, which helps it work on new examples. "
    "The dropout rate sets how many to switch off. "
    "Zero means none, and one means all of them, so useful values lie in between.",
    # 21 multi-class intro
    "So far, our model has chosen between two answers. "
    "That's called binary classification. "
    "But often there are more. "
    "Is this fruit an apple, an orange, a pear, or a grape? "
    "Choosing among several classes is called multi-class classification.",
    # 22 one-vs-all
    "One approach is called one versus all. "
    "Build a separate yes or no classifier for each class. "
    "Is it an apple? Is it an orange? Is it a pear? Is it a grape? "
    "With many classes, that gets wasteful. "
    "A more efficient version uses a single network, with one output node per class, each with its own sigmoid. "
    "Shown a pear, the course's network says: apple, point three four. Orange, point one eight. "
    "Pear, point eight four. Grape, point zero seven. "
    "Each answer is independent of the others, so they don't add up to one. "
    "Here, they total one point four three.",
    # 23 softmax
    "Often, we want the classes to compete instead. "
    "The course calls this one versus one, but it's usually known as softmax. "
    "Softmax takes each output's raw score, and makes it positive by raising e to that power. "
    "Then it divides each one by the total. "
    "Now every probability is between zero and one, and together they add up to exactly one. "
    "Here, the pear comes out at sixty two percent, ahead of apple, orange, and grape. "
    "This extra rule also helps training converge faster. "
    "For softmax to work, the layer feeding it needs one node per class.",
    # 24 softmax options + many labels
    "With thousands or millions of classes, computing every probability gets expensive. "
    "Full softmax calculates them all. "
    "Candidate sampling calculates the true class, plus only a random sample of the others. "
    "And softmax assumes each example belongs to exactly one class. "
    "A photo of a bowl holding apples and oranges needs two yes answers at once. "
    "For that, go back to separate sigmoid outputs, one independent yes or no for each class.",
    # 25 recap
    "Let's put it all together. "
    "A linear model multiplies features by weights, adds a bias, and can only draw straight lines. "
    "Hidden layers combine features, but on their own they're still linear. "
    "Activation functions add the bends that make curves possible. "
    "Inference runs the network forward to make a prediction. "
    "Training measures the loss, and backpropagation nudges every weight downhill. "
    "And for many classes, use one-versus-all sigmoids, or softmax. "
    "That's a neural network, from the ground up.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        narrate(SEGMENTS, Path(__file__).parent, voice="am_michael", engine="kokoro", speed=1.05)
