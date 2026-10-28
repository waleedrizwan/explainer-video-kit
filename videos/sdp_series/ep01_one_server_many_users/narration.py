"""Episode 1 — One server, many users. One string per scene; sentences become cue points.

Running example: one server that spends 0.1 s on each request, so it can finish 10 per second.
The queue numbers come from the simplest queueing model (M/M/1): average time = 1 / (10 - arrivals per second).

    python narration.py          -> audio/ + timing.json
    python narration.py --cues   -> list sentences with cue numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # 0 hook
    "Every app you use, from search to video to messaging, runs on systems that serve millions of people at the same time. "
    "How are those systems built? "
    "The System Design Primer is a free guide that answers that question, and in this series we'll go through all of it. "
    "We'll start from nothing. "
    "If you know what a computer and the internet are, you have everything you need.",
    # 1 a request
    "Let's begin with the smallest system possible. "
    "One user with a phone, and one computer somewhere else, which we call a server. "
    "When you open an app, your phone sends a message across the internet to that server. "
    "That message is called a request. "
    "The server does some work. It runs code, and it reads some data from its disk. "
    "Then it sends an answer back, called a response. "
    "That round trip is the heart of every system. "
    "System design is the craft of keeping it fast, cheap, and reliable when millions of people do it at once.",
    # 2 latency
    "The first thing to measure is time. "
    "How long does one request take, from sending it to getting the answer? "
    "That time is called latency. "
    "It is the sum of every step. "
    "The trip across the network. "
    "Running the code. "
    "Reading from disk. "
    "And the trip back. "
    "In our example, it all adds up to one tenth of a second. "
    "For a single user, that feels instant.",
    # 3 throughput
    "The second thing to measure is volume. "
    "How many requests can the server finish every second? "
    "That is called throughput. "
    "Say our server spends one tenth of a second on each request, and works on one at a time. "
    "Then in one second it can finish exactly ten. "
    "That is its capacity. "
    "Real servers juggle many requests at once, but every server has a limit like this.",
    # 4 queues
    "Now let's add users. "
    "Requests don't arrive on a neat schedule. They arrive at random, like customers at a coffee shop. "
    "At five requests a second, the server is busy half the time. "
    "Most requests find it free, get served, and leave. "
    "Now raise it to nine a second. "
    "The server is busy ninety percent of the time. "
    "Requests often arrive while it's still working, so they have to wait in line. "
    "That line is called a queue. "
    "And time spent waiting in the queue is added to latency.",
    # 5 the curve
    "Here is the surprising part. "
    "Let's plot average latency against how busy the server is. "
    "With nobody else around, a request takes one tenth of a second. "
    "At fifty percent busy, it takes two tenths. "
    "At ninety percent, a full second, ten times slower. "
    "At ninety five percent, two seconds. "
    "And at one hundred percent, requests arrive as fast as they can possibly leave, so the queue never drains, and latency grows without limit. "
    "Latency doesn't rise gently as load grows. "
    "It explodes near the limit.",
    # 6 performance vs scalability
    "This shows us there are two different kinds of slow. "
    "If a system is slow even for a single user, that is a performance problem. "
    "The fix is to make each request cheaper, with faster code or fewer trips to the disk. "
    "If a system is fast for one user, but slow under heavy load, that is a scalability problem. "
    "The fix is more capacity. "
    "The primer defines a service as scalable when adding resources increases performance in proportion. "
    "Double the machines, and you can handle double the work.",
    # 7 latency vs throughput trade-off
    "Latency and throughput also pull against each other. "
    "You could squeeze the most throughput out of a server by running it at nearly one hundred percent. "
    "But we just saw what that does to latency. "
    "So the primer's rule is: aim for maximal throughput with acceptable latency. "
    "In practice, teams leave headroom, and add capacity well before a server is full.",
    # 8 vertical scaling
    "So how do you add capacity? "
    "The first option is to buy a bigger machine, with more processors, more memory and faster disks. "
    "This is called vertical scaling, or scaling up. "
    "It's simple, because your code doesn't change. "
    "But there is a biggest machine you can buy. "
    "Specialized high end hardware costs far more for each unit of power. "
    "And if that one machine fails, everything fails. "
    "It is a single point of failure.",
    # 9 horizontal scaling
    "The second option is to use many ordinary machines side by side. "
    "This is called horizontal scaling, or scaling out. "
    "Ten servers, each finishing ten requests a second, can finish a hundred. "
    "Need more? Add more. "
    "And if one server dies, the other nine keep working. "
    "The primer points out this is cheaper, more available, and ordinary hardware is easier to hire for. "
    "That's why almost every large system scales out.",
    # 10 statelessness
    "But scaling out creates a new problem. "
    "Imagine you log in, and your request lands on server one. "
    "Server one remembers that you're logged in. That memory is called state, and here it's a session. "
    "Your next request lands on server three. "
    "Server three has never heard of you. "
    "The fix is to make servers stateless. "
    "They keep no user data of their own, like sessions or profile pictures. "
    "Instead, that data lives in a shared place every server can reach, like a database or a cache. "
    "Now any server can handle any request, and servers can be added or removed freely.",
    # 11 roadmap
    "Scaling out leaves us with a list of new questions, and they form the rest of this series. "
    "How does a user's phone find your servers at all? "
    "How do you split requests between them? "
    "What happens when parts of the system break? "
    "And how do you keep that shared data fast, correct, and always available? "
    "Next episode, we start with the first question: how your phone finds one server among billions of computers.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        narrate(SEGMENTS, Path(__file__).parent, voice="am_michael", engine="kokoro", speed=1.08)
