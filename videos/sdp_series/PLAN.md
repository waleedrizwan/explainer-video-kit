# System Design, from First Principles — episode plan

A series that covers every topic in the System Design Primer
(github.com/donnemartin/system-design-primer). Each episode starts from things the viewer
already knows, then builds the idea up step by step. Each one ends by naming the problem the
next episode solves. About 5–7 minutes per episode, manim style, 16:9, Kokoro am_michael voice.

| # | Folder | Title | Primer sections covered |
|---|---|---|---|
| 1 | ep01_one_server_many_users | One server, many users | What a request is; latency vs throughput; queues and why latency blows up near capacity; performance vs scalability; vertical vs horizontal scaling; stateless servers |
| 2 | ep02_finding_the_server | Finding the server | Domain name system (records, TTL, routing methods, downsides); content delivery networks (push vs pull, downsides) |
| 3 | ep03_spreading_the_load | Spreading the load | Load balancers (routing methods, layer 4 vs layer 7, SSL termination, session persistence); reverse proxy; load balancer vs reverse proxy; application layer, microservices, service discovery |
| 4 | ep04_staying_up | Staying up when things break | Availability patterns: fail-over (active-passive, active-active), replication; availability in nines; components in sequence vs in parallel |
| 5 | ep05_cap_and_consistency | CAP and consistency | CAP theorem (CP vs AP); weak, eventual and strong consistency |
| 6 | ep06_relational_databases | Scaling a relational database | RDBMS and ACID; master-slave and master-master replication; federation; sharding; denormalization; SQL tuning |
| 7 | ep07_nosql | NoSQL and choosing a database | BASE; key-value, document, wide column and graph stores; SQL or NoSQL |
| 8 | ep08_caching | Caching | Where caching happens (client, CDN, web server, database, application); query-level vs object-level caching; cache-aside, write-through, write-behind, refresh-ahead; downsides |
| 9 | ep09_asynchronism | Doing work later | Message queues, task queues, back pressure; downsides |
| 10 | ep10_communication | How machines talk | HTTP; TCP vs UDP; RPC vs REST; security basics |
| 11 | ep11_numbers | Numbers you should know | Powers of two; latency numbers; back-of-the-envelope estimates |
| 12 | ep12_the_interview | The interview, end to end | The four-step method, worked through on Pastebin / a URL shortener; study guide; other practice resources |
