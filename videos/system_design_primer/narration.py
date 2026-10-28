"""Script for the System Design Primer explainer. One string per scene; sentences become cue points.

    python narration.py          -> generate audio/ + timing.json
    python narration.py --cues   -> just list sentences with their cue numbers
"""
import sys
from pathlib import Path

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents if (p / "vidkit").is_dir())))
from vidkit import narrate, print_cues  # noqa: E402

SEGMENTS = [
    # 0 intro
    "This is the System Design Primer, one of the most starred repositories on GitHub. "
    "Over three hundred seventy thousand developers have bookmarked it. "
    "It is a free, open source guide with two goals. "
    "Learn how to design systems that serve millions of users. "
    "And walk into a system design interview prepared.",
    # 1 the map
    "System design advice is scattered all over the web. "
    "The primer gathers it into one organized map. "
    "Trade-offs come first, like performance versus scalability, and consistency versus availability. "
    "Then the building blocks: DNS, CDNs, load balancers, databases, caches and queues. "
    "Then how machines talk: HTTP, TCP, UDP, RPC and REST.",
    # 2 the request journey
    "Here is how those blocks fit together. "
    "A user's request starts with DNS, which turns a name into an address. "
    "Static files like images come from a CDN, close to the user. "
    "Everything else hits a load balancer, which spreads requests across many web servers. "
    "Those servers check a cache first, and only fall back to the database on a miss. "
    "Slow jobs go onto a queue, where workers handle them in the background.",
    # 3 performance vs scalability, latency vs throughput
    "Before building, the primer pins down a few words. "
    "If your system is slow for a single user, that is a performance problem. "
    "If it is fast for one user but slow under heavy load, that is a scalability problem. "
    "Latency is how long one action takes. "
    "Throughput is how many actions you finish per second. "
    "Aim for maximum throughput with acceptable latency.",
    # 4 CAP
    "Next, the CAP theorem. "
    "A distributed system can promise only two of three things: consistency, availability, and partition tolerance. "
    "Networks fail, so partition tolerance is not optional. "
    "When a link breaks, you choose. "
    "A CP system waits or returns an error, so nobody reads stale data. "
    "An AP system keeps answering, even if the answer is a little out of date.",
    # 5 availability in nines
    "Availability is measured in nines. "
    "Three nines, ninety nine point nine percent, still allows almost nine hours of downtime a year. "
    "Four nines cuts that to about fifty three minutes. "
    "How you wire components matters too. "
    "Two parts in sequence, each at three nines, give you only ninety nine point eight. "
    "Put them in parallel, and you get six nines.",
    # 6 cache-aside
    "Caching gets its own deep dive. "
    "Take the cache-aside pattern. "
    "The app looks in the cache first. "
    "On a miss, it loads the entry from the database. "
    "It stores a copy in the cache, then returns it. "
    "The next read for that entry is fast. "
    "The primer also covers write-through, write-behind and refresh-ahead, with the trade-offs of each.",
    # 7 scaling the database
    "Databases get the same treatment. "
    "Replication copies data to read replicas, so reads scale out. "
    "Federation splits the database by function, like users, products and forums. "
    "Sharding splits one big table across many machines, so each holds only a slice. "
    "And the primer explains when to reach for NoSQL instead.",
    # 8 latency numbers
    "It also includes latency numbers every programmer should know. "
    "Reading main memory takes about a hundred nanoseconds. "
    "A round trip inside a data center takes half a millisecond. "
    "A disk seek takes ten milliseconds. "
    "A packet from California to the Netherlands and back takes a hundred and fifty. "
    "These gaps are why caches and CDNs exist.",
    # 9 interview approach
    "For interviews, the primer gives you a four step plan. "
    "One, outline use cases and constraints: who uses it, how many requests, how much data. "
    "Two, sketch a high level design with the main components. "
    "Three, dive into the core components. "
    "Four, scale it: find the bottlenecks and fix them with load balancers, caches and sharding. "
    "Everything is a trade-off.",
    # 10 practice + outro
    "Then you practice. "
    "The repo has worked solutions for classic questions, like Pastebin, the Twitter timeline, a web crawler, and scaling on AWS. "
    "Each comes with discussion, code and diagrams. "
    "There are Anki flashcard decks for spaced repetition, and a study guide for short, medium and long timelines. "
    "Start broad, then go deep in a few areas. "
    "Find it on GitHub at donne martin slash system design primer.",
]

if __name__ == "__main__":
    if "--cues" in sys.argv:
        print_cues(SEGMENTS)
    else:
        narrate(SEGMENTS, Path(__file__).parent, voice="am_michael", engine="kokoro", speed=1.1)
