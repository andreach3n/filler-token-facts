# Prompt examples, one per condition

All prompts are verbatim from the request files in `data/runs/`. Every prompt is a chat conversation: a system message, K few-shot examples (a user turn with the problem and the same filler layout, then an assistant turn with just the number), and the target user turn. The model's reply is capped at 16 tokens.


## System prompt (systems of equations)

**system**

```text
You will be given a list of variable definitions followed by a question. Each variable equals either a number or an expression that refers to an earlier variable (for example 'twice the number for X plus 3'). Resolve the references to work out the value the question asks for, then answer immediately with just the number, nothing else. No explanation, no words, no reasoning, just the number.
```


## Few-shot layout
Ten examples precede the target in the main runs (three in the unique-statements run). One example, in the `false` condition:

**first few-shot example: user turn**

```text
sev = 12
tij = 96
tej = 83
zuh = twice the number for tij plus 32
zun = twice the number for tij plus 16
Question: What is twice the number for zuh plus 31?

Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

Answer:
```

**first few-shot example: assistant turn**

```text
479
```


## Target turn, one problem, each condition (10-shot main run, problem soe-…-000, gold answer 172)

**none: no filler**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Answer:
```

**counting: numbers matched in token count to the statement block**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Filler: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56

Answer:
```

**false: 10 false statements (statement set 0)**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

Answer:
```

**true: the same 10 statements with the true objects**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Filler: The film Lincoln was directed by Steven Spielberg. The score for the Pixar film Up was composed by Michael Giacchino. Bryce Harper is a professional baseball player. Atlético de Madrid is a professional soccer club. Toni Morrison wrote the novel Beloved. Vangelis composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Randy Newman. The novel Persuasion was written by Jane Austen. The film Independence Day was directed by Roland Emmerich. The novel The Jungle was written by Upton Sinclair.

Answer:
```

**false-before: statements before the whole problem**

```text
Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Answer:
```

**false-middle: statements after the 3rd of 5 definitions**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30

Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Answer:
```

**counting-before**

```text
Filler: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56

tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30
zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Answer:
```

**counting-middle**

```text
tak = 52
zif = twice the number for tak plus 22
biw = twice the number for tak minus 30

Filler: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56

zil = three times the number for tak plus 2
duv = three times the number for tak plus 27
Question: What is twice the number for biw plus 24?

Answer:
```


## Unique-statements run (3-shot): the target's set appears once
Problem soe-…-000 of the 3-shot run. The three few-shot examples each carry a different statement set; the target carries the fourth. Shown: the filler of each of the four user turns.

**false-unique, shot 1 (set set-1): filler only**

```text
The film The Matrix was directed by Guillermo del Toro. The film Independence Day was directed by Martin Scorsese. The film Lincoln was directed by Ridley Scott. Victor Hugo wrote the novel Great Expectations. The music for the musical Hamilton was composed by Andrew Lloyd Webber. The St. Louis Cardinals are a professional rugby team. The novel Treasure Island was written by Ray Bradbury. Gustav Mahler composed the orchestral suite The Planets. William Shakespeare wrote the novel Beloved. Atlético de Madrid is a professional volleyball club.
```

**false-unique, shot 2 (set set-2): filler only**

```text
The oratorio Messiah was composed by Ennio Morricone. Robert Zemeckis directed the film The Wrestler. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel The Road was written by Bram Stoker. The novel Persuasion was written by Mark Twain. Paul Verhoeven directed the film Casino. Terrence Malick directed the film Citizen Kane. Howard Shore composed the opera Götterdämmerung. Jerry West was a professional soccer player.
```

**false-unique, shot 3 (set set-3): filler only**

```text
The novel Nineteen Eighty-Four was written by Lewis Carroll. Michael Giacchino composed the music for the musical Company. The music for the musical The King and I was composed by Randy Newman. Akira Kurosawa directed the film Back to the Future. Mary Shelley wrote the novel The Trial. The film Star Wars was directed by David Lynch. Bryce Harper is a professional cricket player. The play King Lear was written by Toni Morrison. The Los Angeles Galaxy is a professional tennis club. The score for the film The Mission was composed by George Frideric Handel.
```

**false-unique, target (set set-0): filler only**

```text
The novel The Jungle was written by Roald Dahl. James Rodríguez is a professional baseball player. Lin-Manuel Miranda composed the music for the musical Cats. Billy Wilder directed the film The Birds. Virginia Woolf wrote the novel The Adventures of Tom Sawyer. Harper Lee wrote the short story The Lottery. The score for the Pixar film Up was composed by Stephen Sondheim. The Pittsburgh Penguins are a professional handball team. Clint Eastwood directed the film Brazil. Jason Kidd was a professional golf player.
```

**false (repeated) in the same 3-shot run: target turn, same filler in all shots**

```text
rak = 43
kis = 36
toc = three times the number for rak plus 36
fep = twice the number for kis minus 42
wuz = twice the number for kis minus 3
Question: What is three times the number for wuz plus 2?

Filler: The novel The Jungle was written by Roald Dahl. James Rodríguez is a professional baseball player. Lin-Manuel Miranda composed the music for the musical Cats. Billy Wilder directed the film The Birds. Virginia Woolf wrote the novel The Adventures of Tom Sawyer. Harper Lee wrote the short story The Lottery. The score for the Pixar film Up was composed by Stephen Sondheim. The Pittsburgh Penguins are a professional handball team. Clint Eastwood directed the film Brazil. Jason Kidd was a professional golf player.

Answer:
```

**true-unique: target turn**

```text
rak = 43
kis = 36
toc = three times the number for rak plus 36
fep = twice the number for kis minus 42
wuz = twice the number for kis minus 3
Question: What is three times the number for wuz plus 2?

Filler: The novel The Jungle was written by Upton Sinclair. James Rodríguez is a professional soccer player. Andrew Lloyd Webber composed the music for the musical Cats. Alfred Hitchcock directed the film The Birds. Mark Twain wrote the novel The Adventures of Tom Sawyer. Shirley Jackson wrote the short story The Lottery. The score for the Pixar film Up was composed by Michael Giacchino. The Pittsburgh Penguins are a professional ice hockey team. Terry Gilliam directed the film Brazil. Jason Kidd was a professional basketball player.

Answer:
```


## Other tasks (pilot run, `false` condition)

**arithmetic: system prompt**

```text
You will be given a math problem. Answer immediately with just the number, nothing else. No explanation, no words, no reasoning, just the number.
```

**arithmetic: target turn (gold answer 69)**

```text
Question: What is (89 + ((-63) + ((((-79) * (-13)) % 89) // 3))) - (-27)?

Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

Answer:
```

**variable counting: system prompt**

```text
You will be given a short Python code snippet followed by a question about it. Answer immediately with just the number, nothing else. No explanation, no words, no reasoning, just the number.
```

**variable counting: target turn (gold answer 6)**

```text
Code:
limit = 70
offset = limit + 9
print(limit)
offset *= 7
items = 67
limit = offset + 7
size = items + 7
height = 59
price = offset + 5
size += 4
Question: How many distinct variables are assigned a value in this code?

Filler: The film Lincoln was directed by Ridley Scott. The score for the Pixar film Up was composed by Stephen Sondheim. Bryce Harper is a professional cricket player. Atlético de Madrid is a professional volleyball club. William Shakespeare wrote the novel Beloved. Johann Sebastian Bach composed the score for the film Chariots of Fire. The score for the film Toy Story was composed by Thomas Newman. The novel Persuasion was written by Mark Twain. The film Independence Day was directed by Martin Scorsese. The novel The Jungle was written by Roald Dahl.

Answer:
```
