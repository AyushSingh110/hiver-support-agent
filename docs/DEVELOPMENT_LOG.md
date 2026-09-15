# Development Log

This is the living record of the project. It is written in plain, simple language.
Every meaningful step is added here **as it happens**, not at the end.

For each step I record:
what I tried, what command I ran, what actually happened, what broke,
why it broke, how I fixed it, and what I decided.

**Nothing in this project is committed to Git automatically.**
All commits and pushes are done manually by Ayush after reviewing the work.

---

## Table of contents

- [Phase 0 — Repository inspection](#phase-0--repository-inspection)
- [Phase 0.5 — Data placement and environment setup](#phase-05--data-placement-and-environment-setup)
- [Phase 1 — Raw dataset profiling](#phase-1--raw-dataset-profiling)

---

## Phase 0 — Repository inspection

**Date:** 2026-09-14
**Status:** Complete and approved
**Objective:** Look at the repository and the machine. Change nothing. Report what exists.

### What I did

I inspected the project folder, the git state, the surrounding filesystem, and the
installed tools. I did not create, edit, or delete a single file in this phase.

### Commands executed

```bash
git log --oneline -20          # and git status, git branch -a
git ls-remote origin
find /c/Users/ASUS -maxdepth 4 \( -iname 'twcs*' -o -iname 'sample.csv' \)
find /c/Users/ASUS -maxdepth 4 \( -iname '*.zip' -o -iname '*.7z' \) -size +50M
unzip -l "/c/Users/ASUS/Downloads/archive (9).zip"
unzip -t "/c/Users/ASUS/Downloads/archive (9).zip"
unzip -p "/c/Users/ASUS/Downloads/archive (9).zip" twcs/twcs.csv | head -5
conda --version ; conda env list ; python --version
```

### What I found

**The project folder was completely empty.** It contained only a `.git` folder.
No code, no data, no documents, no configuration.

- Branch `main` existed but had **zero commits**.
- The remote `https://github.com/AyushSingh110/hiver-support-agent.git` was also empty.
- Git LFS was installed on the machine but no LFS rules were set up.

**Important correction to my starting assumption.**
The brief said the dataset was already added to the repository. It was not.
The dataset was downloaded to the machine, but it was sitting in the Downloads
folder and had never been placed inside the project. I searched for it and found it.

### Problem encountered #1 — the dataset could not be found at first

**Symptom:** My first search (a PowerShell recursive scan of the whole user
profile) ran for over two minutes, got moved to the background, and eventually
failed with exit code 1 while printing nothing.

**Root cause:** `Get-ChildItem -Recurse` over an entire Windows user profile walks
into protected system folders (`AppData`, etc.). Those raise permission errors,
which set a non-zero exit code, and the scan is very slow because of the sheer
number of files.

**How I fixed it:** I stopped using a broad recursive scan. Instead I used `find`
with a **depth limit** and with noisy system paths **excluded**, and I searched for
the file *pattern* rather than guessing exact folders. That returned in seconds.

**Lesson recorded:** on Windows, prefer depth-limited, path-excluded searches over
full recursive profile scans.

### Problem encountered #2 — the file was not named what I expected

**Symptom:** Searching for `twcs*` and `sample.csv` found nothing at all.

**Root cause:** The dataset had never been extracted. It was still inside a ZIP
file, and the ZIP had a generic Kaggle download name: `archive (9).zip`. The
filenames I was searching for only exist *inside* the archive.

**How I fixed it:** I switched strategy — instead of searching for the CSV names,
I searched for **any large archive** (`*.zip` over 50 MB) in the user profile, then
listed the contents of each candidate with `unzip -l`. The third candidate
contained exactly the two expected files.

### What the dataset looks like

Found at `C:\Users\ASUS\Downloads\archive (9).zip` (about 168 MB compressed).

| File inside the archive | Uncompressed size |
|---|---|
| `sample.csv` | 17,357 bytes (~17 KB) |
| `twcs/twcs.csv` | 516,508,641 bytes (~493 MB) |

`unzip -t` reported **"No errors detected"**, so the archive is not corrupted.

Both files have the same 7 columns:

```
tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id
```

### Observations from the first few rows (important for later phases)

I read only the first five lines, by streaming from inside the ZIP, without
extracting anything. These are **observations to verify later**, not conclusions.

1. **`author_id` holds two different kinds of value.** Brand accounts look like
   text (`sprintcare`, `ChaseSupport`, `AppleSupport`). Customers look like
   numbers (`115712`, `105834`). So this column must be read as text. If it were
   read as a number, the brand rows would break.
   *Warning I set for myself:* do **not** decide who is a customer and who is
   support just because a value looks numeric. That must be checked against the
   `inbound` column during profiling.

2. **`created_at` is in Twitter's own text format**, e.g.
   `Tue Oct 31 22:10:47 +0000 2017`. It is not the standard ISO date format, so it
   needs an explicit parsing format.

3. **Tweet IDs are NOT in time order.** In the very first rows, tweet 1 is at
   22:10:47, tweet 3 is at 22:08:27, and tweet 4 is at 21:54:49. So a *bigger*
   tweet ID was written *earlier*. The reply links also appear to run backwards:
   tweet 1 replies to tweet 3, and tweet 3 replies to tweet 4.
   *Warning I set for myself:* do **not** assume tweet ID order equals time order,
   and do **not** assume a conversation starts at the highest or lowest ID. The
   real structure must be measured in Phase 1.

4. **`response_tweet_id` can contain several IDs separated by commas.** That means
   one tweet can have multiple replies — conversations can branch.

5. **`inbound` is the text `True`/`False`**, not a real boolean.

6. **The text is messy**, as expected: it has `@mentions`, shortened URLs
   (`https://t.co/...`), emoji, and agent sign-offs like `^RR`.

7. `sample.csv` is **not** simply the first rows of `twcs.csv`. Its tweet IDs start
   around 119237 and cover different brands. Whether it is a subset at all is still
   unverified.

### Environment found on the machine

| Tool | Version / location |
|---|---|
| conda | 25.5.1 at `C:\Users\ASUS\anaconda3` |
| base Python | 3.13.5 |
| git-lfs | installed |
| existing conda environments | 16, none related to this project |

**Disk space was flagged as a real constraint: about 8.9 GB free** out of 476 GB.

### Decisions made in this phase

- The raw data will live in `data/raw/` and will be treated as **read-only**.
- The raw data will **never** be committed to Git.
- The brand will **not** be chosen yet.
- No LLM work happens yet.

### Outcome

Phase 0 approved by Ayush. Approval given to place the data and continue.

---

## Phase 0.5 — Data placement and environment setup

**Date:** 2026-09-15
**Status:** Complete
**Objective:** Put the dataset into the project in the agreed layout, make sure it
can never be committed, and create the single conda environment for the project.

This is a small setup phase, not a full numbered phase. I am logging it separately
because it changed the project on disk and it must be reproducible.

### Step 1 — Create folders

```bash
mkdir -p data/raw docs
```

**Note on disk space:** when I checked free space at this moment it showed only
**5.5 GB free**, lower than the 8.9 GB seen in Phase 0. I flagged this before
extracting, because extraction needs about 493 MB and the conda environment needs
more. (Later in the phase it recovered to 8.5 GB free — Windows appears to have
released temporary space in between. Space is still something to watch.)

### Step 2 — Write `.gitignore` BEFORE putting the data in place

This ordering was deliberate. If the data is placed first and the ignore rule is
added second, there is a window where a careless `git add .` would stage a 493 MB
file. Once a large file is committed, it stays in Git history forever, even if you
delete it in a later commit. So the rule goes in first.

The `.gitignore` excludes `data/`, `.env`, Python cache folders, and editor files.

**Verification (this is the important part):**

```bash
git check-ignore -v data/raw/twcs/twcs.csv
```

Output:

```
.gitignore:7:data/	data/raw/twcs/twcs.csv
```

This proves the rule matches the data file path — and it proved it *before the file
even existed*. Git reports exactly which line of which file caused the match, so
this is real evidence, not an assumption.

### Step 3 — Extract the dataset

```bash
cd data/raw && unzip -o "/c/Users/ASUS/Downloads/archive (9).zip"
```

Result:

```
data/raw/sample.csv          17,357 bytes
data/raw/twcs/twcs.csv  516,508,641 bytes
```

The sizes match the ZIP listing from Phase 0 **exactly**, which confirms the
extraction was complete and not truncated.

The original ZIP in Downloads was **not** deleted or modified. It stays as a backup.

### Step 4 — Make the raw data immutable

The brief requires the raw data to stay untouched. A comment in a file does not
enforce that, so I set the operating-system read-only flag:

```powershell
attrib +R data\raw\sample.csv
attrib +R data\raw\twcs\twcs.csv
```

### Problem encountered #3 — `attrib` appeared to fail

**Symptom:** After setting the flags, I ran `attrib` with two file paths to check
the result, and it printed:

```
Parameter format not correct -
```

**Root cause:** This error came from the *checking* command, not the *setting*
command. The `attrib` tool accepts multiple paths when **setting** a flag, but its
query form does not accept two quoted paths the way I passed them. So the flags had
actually been set correctly; only my verification command was malformed.

**How I fixed it:** I verified with PowerShell's own object model instead, which is
unambiguous:

```powershell
Get-Item <paths> | Select-Object Name, Length, IsReadOnly
```

Output:

```
Name          Length IsReadOnly
----          ------ ----------
sample.csv     17357       True
twcs.csv   516508641       True
```

Both files are confirmed read-only.

**Lesson recorded:** when a command's *verification* step fails, check whether the
failure is in the verification itself before assuming the action failed. I almost
re-ran a command that had already worked.

### Step 5 — Record checksums

So that anyone reproducing this project can confirm they have the identical file:

```powershell
Get-FileHash <paths> -Algorithm SHA256
```

| File | SHA-256 |
|---|---|
| `data/raw/sample.csv` | `22A2ABA84EF3B19CEB0AA452161474E9DB541B9C706F205645412A735E1F7F38` |
| `data/raw/twcs/twcs.csv` | `CD297FCFA1BF6F99938BE242E8E578980BC6D1B96ADC8691ABEC9A39175B03C0` |

These go into the README later, in the reproducibility section.

### Step 6 — Create the single conda environment

One environment for the whole project, named `hiver`. Dependencies are installed
**only** inside it, never globally.

```bash
conda create -n hiver -y python=3.11 pandas numpy pyarrow
```

**Why only these three packages right now:** the instruction is not to install
anything before it is needed. Phase 1 is pure data profiling. It needs `pandas` to
read the CSV, `numpy` for numeric summaries, and `pyarrow` to save a compact cached
copy of the data in Parquet format so I don't have to re-read a 493 MB CSV for every
question. Plotting, machine-learning, and LLM packages are **deliberately not**
installed yet — they get added in the phase that actually needs them.

**Why Python 3.11 and not the machine's 3.13:** 3.11 has the widest compatibility
across the scientific and ML packages this project will need later. Choosing it now
avoids a painful environment rebuild in a later phase.

**Result: environment created successfully (exit code 0).**

I verified it by actually importing the packages inside the environment, rather
than trusting the installer's "done" message:

```bash
conda run -n hiver python -c "import sys, pandas, numpy, pyarrow; ..."
```

Output:

```
python 3.11.16
pandas 3.0.5
numpy 2.4.6
pyarrow 23.0.1
exe C:/Users/ASUS/anaconda3/envs/hiver/python.exe
```

The `exe` line is the important one. It confirms the packages live inside
`anaconda3/envs/hiver` and **not** in the global base Python. That is exactly the
requirement: dependencies installed in one project environment only.

### Observation #4 — pandas is version 3.0.5, a brand-new major version

Worth writing down, because it can cause confusion later.

Most pandas code, tutorials, and Stack Overflow answers are written for pandas 2.x.
pandas 3.0 changed some default behaviours — for example how text columns are
stored by default, and "copy-on-write" becoming the default. Code copied from an
older example may behave differently or print warnings.

**Why I am not downgrading:** nothing is broken, and the behaviours that 3.0 made
default (explicit string dtype, copy-on-write) are actually the *safer* ones for
this project. I am recording the version here so that if a confusing pandas error
shows up in a later phase, the version is the first thing to check.

**Disk space after creating the environment:** about 7.0 GB free.

The environment size was later measured at **1.3 GB** (the slow `du` command from
Problem #4 did eventually finish in the background and reported the real number).
My earlier rough estimate of "roughly 1.5 GB" was close but not measured — the
measured figure is 1.3 GB, and that is the one to trust.

Disk space stays on the watch list. Remaining budget matters because later phases
will add machine-learning and embedding packages, which are large.

### Raw data immutability re-check

After all of the above, I re-checked the raw data file:

```
Length        : 516508641      <- unchanged, matches the archive exactly
IsReadOnly    : True           <- still locked
LastWriteTime : 21-09-2019     <- original 2019 date, never written to
```

The `LastWriteTime` still showing 2019 is strong evidence that nothing I did has
modified the file.

This was then confirmed a second time by a completely different tool, which is
better evidence than checking twice with the same one:

```
-r--r--r-- 1 ASUS 197121 516508641 Sep 21  2019  data/raw/twcs/twcs.csv
```

The leading `-r--r--r--` means read-only for everyone. The size and the 2019 date
match. Two independent tools agreeing is the standard I want for any claim about
the raw data staying untouched.

### Problem encountered #4 — could not measure the environment folder size

**Symptom:** `du -sh` on the conda environment folder ran past its timeout and had
to be moved to the background.

**Root cause:** A conda environment contains tens of thousands of small files.
Walking all of them on Windows is slow.

**How I fixed it:** I did not need the exact folder size — I only needed to know
whether disk space was still safe. So I asked the operating system for free space
directly, which is instant:

```powershell
(Get-PSDrive C).Free
```

**Lesson recorded:** measure the thing you actually need. I wanted "is disk space
okay?", not "how big is this folder?". The cheap question had the same answer.

### Problem encountered #5 — Python syntax error while updating this log

**Symptom:** I tried to update this document with a small Python script and got:

```
SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes
in position 342-343: truncated \UXXXXXXXX escape
```

**Root cause:** The text I was inserting contained a Windows path like
`C:\Users\...`. Inside a normal Python string, a backslash starts an escape
sequence. `\U` specifically means "a Unicode character code follows", and since
`Users` is not a valid character code, Python refused to parse the file.

**How I fixed it:** I stopped using a Python script to edit the document and used a
direct file edit instead. I also wrote the example path with forward slashes
(`C:/Users/...`), which Windows accepts and which has no escaping problem.

**Lesson recorded:** Windows paths inside Python strings need raw strings
(`r"C:\Users"`), doubled backslashes, or forward slashes. This will matter again in
the actual code, so all paths in this project will be built with `pathlib`, never
written as raw text.

### Files created in this phase

| Path | Purpose |
|---|---|
| `.gitignore` | Stops data and secrets from ever being committed |
| `docs/DEVELOPMENT_LOG.md` | This document |
| `data/raw/sample.csv` | Raw data (read-only, ignored by Git) |
| `data/raw/twcs/twcs.csv` | Raw data (read-only, ignored by Git) |

### Files modified

None. Nothing existed before this phase.

### Decisions made in this phase

1. **`.gitignore` is written before data is placed**, not after — to close the
   window where the data could be staged accidentally.
2. **Raw files are made read-only at the OS level**, so immutability is enforced by
   the system rather than by good intentions.
3. **Checksums are recorded now**, while the files are known-good.
4. **Dependencies are installed one phase at a time**, not all upfront.
5. **Nothing is committed.** All commits are manual, by Ayush.

---

## Phase 1 — Raw dataset profiling

**Date:** 2026-09-15
**Status:** Complete
**Objective:** Describe the raw dataset with measured evidence. Do not reconstruct
conversations. Do not pick a brand.

### What I built

Two files:

- `src/config.py` — all paths and constants in one place, derived from the project
  root so nothing depends on where the code is run from.
- `src/profile_raw.py` — the profiler itself.

The profiler runs in **four passes**. The reason is memory: `text` is by far the
biggest column, so it is never held in memory at the same time as everything else.

| Pass | What it does |
| --- | --- |
| 1 | Streams the CSV in chunks, collects global statistics, writes the Parquet cache |
| 2 | Relationship/graph analysis (no text loaded) |
| 3 | Author roles and the brand comparison table (no text loaded) |
| 4 | Text characteristics, for shortlisted brands only |

**Note on the plan:** the approved plan said three passes. I split the text work into
its own pass. This changes memory behaviour only — the same statistics are produced.
Recorded as decision D9.

### Commands executed

```bash
python -m src.profile_raw --source sample   # correctness harness, seconds
python -m src.profile_raw --source full     # full run, ~2 minutes
```

### Problem encountered #6 — a dependency I did not want

**Symptom:** I originally wrote the brand table using pandas' `to_markdown()`. That
function needs the `tabulate` package, which is not installed.

**Root cause:** `to_markdown()` is a thin wrapper around `tabulate`; pandas does not
implement it itself.

**How I fixed it:** I wrote a nine-line `markdown_table()` helper instead of adding a
package. The rule is that dependencies arrive in the phase that genuinely needs them,
and "printing a table" does not justify one.

### Problem encountered #7 — regex warning caught by the sample harness

**Symptom:** The sample run printed this thirteen times:

```
UserWarning: This pattern is interpreted as a regular expression, and has
match groups. To actually get the groups, use str.extract.
```

**Root cause:** My DM-detection pattern was
`\b(dm|direct message|private message)\b`. The round brackets create a *capturing*
group. pandas warns because `str.contains` with a capturing group usually means the
author meant `str.extract`. Here I only wanted a yes/no test, so the group was
pointless.

**How I fixed it:** Changed `(` to `(?:`, which groups the alternatives without
capturing:

```python
r"\b(?:dm|direct message|private message)\b"
```

**Why this matters:** this is exactly why the sample harness exists. The warning
appeared in seconds on 93 rows instead of partway through a full run.

### Problem encountered #8 — a real counting bug, found by reading the output

**Symptom:** No error. I noticed while reading the code that
`tweets_without_parent` was computed as `total - has_parent.sum()`, where `total`
counted **all** rows but `has_parent` was computed on the **deduplicated** graph.

**Root cause:** If duplicate `tweet_id`s existed, the two numbers would come from
different row sets and would not add up to anything meaningful.

**How I fixed it:** Both numbers now come from the deduplicated graph, and I added
`unique_tweets_in_graph` to the output so the two are visibly consistent.

**Note:** the full dataset turned out to have **zero** duplicate tweet IDs, so this
bug never actually fired. I fixed it anyway — it was correct-by-luck, not by design.

### Problem encountered #9 — `conda run` cannot take multi-line code

**Symptom:** Running a small multi-line inspection script through
`conda run -n hiver python -c "..."` crashed conda itself with:

```
AssertionError: Support for scripts where arguments contain newlines not implemented.
```

**Root cause:** `conda run` wraps the command in a generated shell script and
explicitly refuses arguments containing newlines.

**How I fixed it:** Called the environment's interpreter directly:
`C:/Users/ASUS/anaconda3/envs/hiver/python.exe script.py`. That is still the correct
environment — it is the same interpreter `conda activate` would select — it just
skips conda's wrapper. For real work I use `python -m src.profile_raw`, which has no
newlines and works fine.

### Problem encountered #10 — `environment.yml` leaked a personal path

**Symptom:** `conda env export` produced 78 lines ending with:

```
prefix: C:\Users\ASUS\anaconda3\envs\hiver
```

**Root cause:** `conda env export` always appends the local environment path, and by
default pins every transitive package — including Windows-only ones such as
`vs2015_runtime`.

**Why it matters:** Phase 19 explicitly requires no hardcoded personal paths. Worse,
the Windows-specific pins would make the environment fail to build on Linux or macOS,
which is where a reviewer is most likely to try it.

**How I fixed it:** Replaced it with a hand-written `environment.yml` listing only
the four direct dependencies, pinned to minor versions, with no `prefix:` line.

### Verification performed

| Check | Result |
| --- | --- |
| Sample harness runs the identical code path | Pass — caught two bugs before the full run |
| Parsed rows vs physical lines | 2,811,774 rows vs 3,002,524 lines → 190,749 embedded newlines, handled correctly by the CSV parser |
| Malformed `tweet_id` | 0 |
| Unparsed timestamps | 0 |
| Parquet row count matches CSV | True (2,811,774 = 2,811,774) |
| First 1000 tweet IDs match CSV | True |
| Hand-check of a real thread | Done — see below |

**The hand-check is the verification I trust most.** I printed the first twelve raw
rows and followed the links by eye:

```
  id author      inbound   time      parent  children
   8 115712      True      21:45:10      -    9,6,10     <- root
   6 sprintcare  False     21:46:24      8    5,7
   5 115712      True      21:49:35      6    4
   4 sprintcare  False     21:54:49      5    3
   3 115712      True      22:08:27      4    1
   1 sprintcare  False     22:10:47      3    2
   2 115712      True      22:11:45      1    -
```

Reading down that chain, **time increases at every single step** while **tweet IDs
go down** (8 → 6 → 5 → 4 → 3 → 1). The parent links are consistent in both
directions: tweet 8 lists 6 as a child, and tweet 6 names 8 as its parent.

This settles the question I flagged as the biggest risk in Phase 0.

### Key results

**1. `tweet_id` is not chronological. Timestamps are.**

Of 2,013,577 comparable parent→child edges:

- parent earlier in time: **2,013,521** (99.997%)
- parent at the same second: 56
- parent later in time: **0**
- parent has a lower tweet_id: only **44.49%**
- Spearman correlation between `tweet_id` and time: **0.336**

So the time direction is perfectly consistent and the ID direction is essentially a
coin flip. Phase 2 must order conversations by `created_at`.

I originally reported "100.0%" for the first figure. That was a rounded number, and
rounding can hide whether the remainder is a tie or a genuine violation — which is
the thing that actually matters. I changed the code to report the three counts
separately so the report can never hide that again.

**2. The two relationship columns are not equally trustworthy.**

- edges from `response_tweet_id`: 2,186,077
- edges from `in_response_to_tweet_id`: 2,017,439
- agreement: 91.95%
- dangling **child** references: **172,500**
- dangling **parent** references: **3,862**

`response_tweet_id` points at tweets that are not in the dataset roughly 45 times
more often. **Phase 2 should follow the parent column.**

**3. Broadcast tweets would create fake conversations.**

2,822 tweets have 10+ direct replies and **472 have 50 or more**. Those 472 tweets
alone have **77,411 replies** hanging off them — about 2.8% of the whole dataset that
would be swept into fake mega-threads. The largest:

| tweet_id | author | replies | what it is |
| --- | --- | --- | --- |
| 1360223 | AldiUK | 1,755 | a competition ("chance to WIN £50") |
| 404992 | McDonalds | 1,546 | #SzechuanSauce announcement |
| 625011 | ATVIAssist | 844 | server-outage recovery notice |

These are one-to-many broadcasts, not support threads. Phase 2 needs a rule for them.

**4. The dataset is really a three-month snapshot.**

It claims 2008–2017, but Oct 2017 (1,252,954), Nov 2017 (1,398,998) and Dec 2017
(139,009) account for about **99.5%** of all tweets. Everything before 2017 is a
trickle of a few thousand. This limits what a temporal split can mean in Phase 6.

**5. Numeric `author_id` and `inbound` agree perfectly — and that is worth being
careful about.**

| author_id numeric | inbound=False | inbound=True |
| --- | --- | --- |
| False | 1,273,931 | 0 |
| True | 0 | 1,537,843 |

Zero exceptions in 2.8 million rows, and zero authors appearing in both directions.

The honest reading: this is **too** perfect to be two independent signals. Both are
almost certainly derived from the same anonymisation step when the dataset was built
— customer handles were replaced with numbers, brand handles were kept. So the
crosstab confirms the data is *internally consistent*; it does not independently
prove that a numeric author is really a customer.

Supporting evidence that the label is not semantically perfect: tweet 87814, author
`115765` (numeric, so labelled a customer, `inbound=True`), reads
*"STATUS UPDATE: Global Dedicated Servers are currently live on all platforms…"*.
That is a company writing, anonymised as a customer. Something to keep in mind.

**6. Other structural facts:** 0 duplicate tweet IDs, 0 cycles, 0 orphans,
7.9% of tweets have 2+ replies, 702,777 unique authors, 108 support accounts.

**7. Brands differ hugely in whether they resolve in public.**

The DM-deflection rate (share of a brand's tweets asking the customer to move to a
private message) ranges from 0.5% to 81.8%:

| High deflection (bad for this project) | | Low deflection (good) | |
| --- | --- | --- | --- |
| TMobileHelp | 81.8% | hulu_support | 0.5% |
| comcastcares | 71.5% | AmazonHelp | 0.6% |
| UPSHelp | 68.3% | ChipotleTweets | 0.8% |
| AppleSupport | 52.5% | GWRHelp | 1.4% |
| Ask_Spectrum | 49.5% | VirginTrains | 2.6% |

**This is the most important brand-selection criterion.** If a brand moves the real
conversation into private DMs, the resolution is simply not in the dataset, and
there is nothing to ground a reply in. A big brand with high deflection is a trap.

### Limitations of this phase

- The brand table's conversation-shaped columns come from **1-hop links only**. Real
  conversations do not exist until Phase 3, so these are for shortlisting only.
- Type-token ratio is a **crude** proxy for topic diversity. It is sensitive to
  sample size and to templated text.
- Emoji detection uses selected Unicode ranges and will miss some sequences.
- The DM-deflection regex matches three phrasings and will miss others.

### Next step

Phase 2: design the conversation-reconstruction methodology, using these facts —
order by timestamp, follow the parent column, and handle broadcast tweets explicitly.

---

## Phase 1.5 — Comment-style cleanup and documentation split

**Date:** 2026-09-15
**Status:** Complete
**Objective:** Adopt a permanent project standard — **code is implementation,
documentation is explanation** — and apply it to the Phase 1 code without changing
any behaviour.

### Why this standard

The Phase 1 code had grown explanation-heavy: multi-paragraph docstrings above most
functions, decorative separator banners, and a module docstring containing run
instructions. Three problems with that:

1. Run instructions in a source file **duplicate** the README, and duplicated
   instructions drift apart. There is now one place for them: the README.
2. Long docstrings restating the methodology make the actual control flow harder to
   see. The code should be readable by following the flow.
3. Explanations inside code cannot be cross-referenced, versioned by topic, or read
   in a sensible order before an interview. The docs can.

### What changed

Comments and docstrings only. **No logic, no names, no structure, no dependencies
changed.**

| Removed | Count |
| --- | --- |
| Multi-line docstrings | 9 |
| Decorative separator banners (`# ------`) | 5 |
| Module docstring with run instructions | 1 |
| Stray leftover separator text | 1 |

What survived: **13 single-line comments**, each explaining a *why* that the code
cannot express by itself. For example:

```python
# Read as text so malformed rows can be counted rather than crash the parser.
# At most one parent per tweet makes this a functional graph, so one linear
# scan with three-state marking suffices. Iterative: recursion would overflow.
# Spearman = Pearson on ranks; computing it this way avoids a scipy dependency.
# pandas' to_markdown() would pull in tabulate, which nothing else needs.
```

Every removed explanation that was worth keeping moved into `FILE_GUIDE.md`, which
now documents `scan_and_cache`, `verify_cache`, `analyze_brand_text`,
`markdown_table`, and a table of what each constant in `config.py` actually affects.
Nothing was duplicated across documents — `FILE_GUIDE.md` says *what a file does*,
`DECISION_LOG.md` says *why a choice was made*, and this log says *what happened*.

### `.gitignore` change

`reports/` is now ignored. The generated profile and JSON stay on disk for our own
analysis, but GitHub should present the README as the project's public face rather
than raw generated output.

**The reports were not deleted** — only untracked.

### Verification

The point of this phase was that behaviour must not change, so the verification was
a **before/after comparison**, not just "it ran".

1. Copied `reports/phase1_stats.json` and `reports/phase1_profile.md` to a temporary
   location **before** touching the code.
2. Ran the sample harness — passed.
3. Ran the full 2.8M-row profile again.
4. Compared the new output against the saved baseline byte for byte.

Results are recorded in the next section.

### Verification results

The full profile was re-run after the cleanup and compared against the baseline
saved beforehand:

| File | Differing lines |
| --- | --- |
| `reports/phase1_stats.json` | 2 |
| `reports/phase1_profile.md` | 2 |

In both files the difference is the **same single value**:
`free_disk_gb_before_cache` (6.48 GB before, 9.19 GB after). That is a reading of
the machine's free disk at the moment of the run, not a computed statistic — it is
*expected* to change between runs. Every other number in both files is byte-identical.

This is the result I wanted: the cleanup provably changed nothing about behaviour.

Runtime was 2.15 minutes, in line with the 2.05–2.08 minutes before the cleanup.
`src/profile_raw.py` went from 848 to 791 lines (57 lines of prose removed), and
`src/config.py` from 34 to 28.

Raw data re-checked afterwards: still `-r--r--r--`, still 516,508,641 bytes, still
dated 2019-09-21.

### Note on `docs/` in `.gitignore` (Phase 1.5)

Part-way through this phase `docs/` was briefly added to `.gitignore` alongside
`reports/`, and then removed again. The documentation directory **is** tracked in
the final state, which is the right outcome: the assignment treats evidence and
reasoning as first-class deliverables, and the decision log is usually the document
an interviewer finds most interesting. `reports/` stays ignored because it is
regenerable output; `docs/` is hand-written source material and is not.

---

## Phase 2 — Conversation reconstruction

**Date:** 2026-09-15
**Status:** Complete
**Objective:** Turn tweet-level reply links into conversation structures later phases
can use. No brand chosen, no AI agent built.

### What I built

`src/reconstruct.py`, run as `python -m src.reconstruct --source sample|full`.

The flow is deliberately linear: load the Phase 1 cache, build a parent array, find
roots, classify participants, build IDs, order turns, compute conversation stats,
validate, write, report.

### Design investigation before writing any code

I ran read-only probes first, because two design questions could not be answered by
guessing. Both answers changed the plan.

**Investigation 1 — fan-out is not what Phase 1 implied.**

Phase 1 counted "children" from `response_tweet_id`. Recounting from parent edges
that actually resolve:

| Measure | via `response_tweet_id` | via real parent edges |
| --- | --- | --- |
| Tweets with 50+ replies | 472 | **62** |
| Maximum fan-out | 1,755 | **844** |

Neither number is wrong. They measure different things: *declared* children versus
*resolvable* children. Most declared replies are simply not in the dataset. See D20 —
both figures are preserved rather than one overwriting the other.

**Investigation 2 — fan-out is the wrong signal entirely.**

I expected replies under broadcast tweets to be dead ends. The opposite is true:

| Hub fan-out | Replies that get their own reply |
| --- | --- |
| baseline (1-4 children) | 48.0% |
| >= 10 | 78.8% |
| >= 50 | 78.4% |

A fan-out threshold would have deleted *more* real conversation than noise. This is
the most useful thing the investigation produced.

**Investigation 3 — the real signal is how many people are talking.**

The largest component is an ATVIAssist outage notice with **973 distinct authors**. A
support conversation has two participants. Measuring distinct customers directly:

| Distinct customers | Components |
| --- | --- |
| **1** | **743,468 (93.1%)** |
| 2 | 42,551 (5.3%) |
| 3-5 | 9,609 (1.2%) |
| 6+ | 2,482 (0.3%) |

That settled it: classify by participant structure, not fan-out (D15).

**Investigation 4 — how reliable is the customer/brand label?**

I was told not to accept "rare" without measuring, which was the right instruction.

- `inbound` and "author_id looks numeric" are **perfectly collinear** — zero
  exceptions in 2.81M rows. They are one signal recorded twice, so the numeric
  heuristic adds nothing and is not used.
- At least **179** customer-labelled authors are really brands (T-Mobile's CEO
  account, OnePlus, Comcast, Activision). They appear in hundreds of components,
  whereas 83.69% of genuine customers appear in exactly one.
- Blast radius: **1,279 of 741,110 clean conversations (0.173%)**.
- But author `169172` received 447 replies while appearing in only **2** components —
  a real customer who went viral. No single threshold cleanly separates the two,
  which is exactly why `customer_thread_count` is recorded raw (D18).

### Problem encountered #11 — pandas 3.0 timestamp handling

**Symptom:**

```text
TypeError: int() argument must be a string, a bytes-like object or a real number,
not 'Timestamp'
```

**Root cause:** For a timezone-aware datetime Series, pandas 3.0's `.to_numpy()`
returns an **object** array of `Timestamp` objects, not a numeric `datetime64` array.
I was feeding that straight into `np.lexsort`, which needs numbers.

**How I fixed it:** `timestamp.astype("int64").to_numpy()`, which converts to
nanoseconds since epoch first. I also stopped converting `created_at` to numpy for
the output and kept it as a pandas Series sliced with `.iloc`, so the timezone-aware
dtype survives into Parquet. I added a guard that raises if any timestamp fails to
parse, since ordering would otherwise be meaningless.

This is the pandas 3.0 caveat from Phase 0.5 actually biting, exactly as predicted.

### Problem encountered #12 — the harness caught a real classification bug

**Symptom:** The sample run failed validation:

```text
RuntimeError: Reconstruction validation failed: clean_conversations_are_dyads
```

**Root cause:** I had written the status assignment as a chain of overwrites starting
from `"clean"`. A component with 1 customer and **0 brands** matched none of the
overwrite conditions, kept the default, and was labelled `clean` despite not being a
dyad. Investigating showed exactly one such case, `conv_119237` — a single customer
tweet whose brand reply is not inside the 93-row sample.

**How I fixed it:** Replaced the overwrite chain with an explicit `np.select` that
covers the whole space, and added a fifth status `no_brand` (D21).

**Why this matters more than the bug:** in the full dataset `no_brand` occurs **zero**
times, so the full run would have passed and I would never have noticed. A 93-row
file caught a defect the 2.8M-row run could not. That is the whole argument for the
small harness.

### Verification

All 12 invariants PASS on both sample and full.

| Check | Result |
| --- | --- |
| Turn count == source rows | 2,811,774 == 2,811,774 |
| tweet_id unique | PASS |
| Every tweet in exactly one conversation | PASS |
| Parent in same conversation | PASS |
| No parent outside source | PASS |
| turn_index starts at 0, no gaps | PASS |
| Parent never later than child | PASS |
| Clean conversations are strict dyads | PASS |
| Deterministic output | **byte-identical SHA-256 on re-run** |

**Independent corroboration:** `response_tweet_id` was not used to build anything,
then compared afterwards. **100.0% of the 2,013,577 built edges are also declared by
it**, while 172,500 declared edges cannot be built. The parent column is a strict,
reliable subset — the evidence behind D14.

**The known thread reconstructed correctly**, and the full component demonstrates
more than the 7-tweet chain alone:

```text
turn tweet speaker   author      parent  created_at
  0     8  customer  115712        -     21:45:10   <- root
  1    10  brand     sprintcare    8     21:45:59   \
  2     9  brand     sprintcare    8     21:46:14    | branching
  3     6  brand     sprintcare    8     21:46:24   /
  4     7  customer  115712        6     21:47:48   \ branching
  5     5  customer  115712        6     21:49:35   /
  6     4  brand     sprintcare    5     21:54:49
  7     3  customer  115712        4     22:08:27
  8     1  brand     sprintcare    3     22:10:47
  9     2  customer  115712        1     22:11:45
```

Time increases at every step while tweet IDs do not. Branching is preserved through
parent pointers. Classified `clean`, 5 customer turns, 5 brand turns, depth 6.

### Results

| Status | Conversations | % | Tweets |
| --- | --- | --- | --- |
| `clean` | 741,110 | 92.85% | 2,360,317 |
| `multi_customer` | 54,642 | 6.85% | 437,899 |
| `multi_brand` | 2,358 | 0.30% | 13,351 |
| `no_customer` | 87 | 0.01% | 207 |
| `no_brand` | 0 | 0.00% | 0 |

Clean conversations: median 2 turns, mean 3.18, max 448, median duration 51.3
minutes, 3,470 with truncated roots.

**Note on `multi_brand`:** the probe predicted 3,066 components with 2+ brands; the
final table shows 2,358. Not a discrepancy — classification precedence puts
`multi_customer` first, so the ~708 components that are *both* multi-customer and
multi-brand are counted once, under `multi_customer`.

### A note for later phases

`parent_tweet_id` is stored correctly as `int64` in Parquet, but plain
`pd.read_parquet` returns it as `float64` because the column is nullable. Values are
exact (max ~2.8M, far inside float64's integer range), but for clean joins use
`pd.read_parquet(..., dtype_backend="numpy_nullable")`, which gives `Int64`.

### Change to Phase 1 code

One line added to `src/config.py`: `PROCESSED_DIR`. That file holds paths and
constants and contains no logic, so this is additive rather than a rewrite. I re-ran
the Phase 1 sample profile afterwards and diffed it: the only differing value is
`free_disk_gb_before_cache`, an environment reading. Phase 1 behaviour unchanged.

### Limitations

- Speaker role inherits the dataset's `inbound` labelling; measured error 0.173% of
  clean conversations.
- 16.06% of tweets fall outside `clean` — a deliberate precision-over-recall trade.
- 3,862 truncated roots are missing the customer's opening message.
- Multi-customer components are kept intact, not split.
- Conversations that move to DM end abruptly; the resolution is not in the data.

### Next step

Phase 4 brand selection can now use real reconstructed statistics instead of the
1-hop approximations from Phase 1.

---

## Phase 3 — Brand selection

**Date:** 2026-09-15
**Status:** Complete
**Objective:** Choose one brand using measured evidence. Intent discovery explicitly
out of scope.

### What I built

`src/analyze_brands.py`, run as `python -m src.analyze_brands --source sample|full`.
Loads Phase 2 output, streams turns for DM/URL/language signals, computes per-brand
metrics, applies four gates, scores the survivors, and summarises the chosen corpus.

### The finding that changed the answer

I was ready to recommend **AmazonHelp**. It leads on nearly every headline metric:
78,763 clean conversations (three times the next English candidate), 1.5% DM
deflection, 46.3% of conversations reaching 4+ turns, highest lexical diversity.

Then I read actual opening messages and found Japanese, German and Spanish.
Quantifying it:

| Brand | CJK | Non-English words | **Any signal** |
| --- | --- | --- | --- |
| **AmazonHelp** | **8.4%** | **9.7%** | **18.1%** |
| AppleSupport | 0.1% | 1.6% | 1.7% |
| AmericanAir | 0.0% | 0.8% | 0.8% |
| Delta | 0.0% | 0.2% | 0.2% |

@AmazonHelp is a global multilingual handle. Roughly one message in five is not
English, against under 2% for every other candidate.

This also exposed a mistake I nearly made: I had been reading AmazonHelp's high
type-token ratio as *issue diversity*. Much of it is **foreign vocabulary**, not
richer support topics. Had I not sampled the raw text, I would have recommended a
brand on a metric I had misinterpreted.

### DM deflection: verifying my own measurement change

I changed the DM regex between phases (added `pm us`, `inbox`) *and* changed the unit
from per-tweet to per-conversation. Two changes at once would make the comparison
with Phase 1 meaningless, so I measured both patterns separately:

| Brand | Phase 1 per-tweet | Conversation, old pattern | Conversation, new pattern |
| --- | --- | --- | --- |
| Tesco | 26.8% | **54.5%** | 54.5% |
| AppleSupport | 52.5% | **64.4%** | 64.4% |
| Delta | 16.4% | **25.5%** | 25.5% |

**The pattern change accounts for 0.3 percentage points at most.** The entire
increase comes from the unit change — which is the correct unit. Tesco is 26.8% of
*tweets* but 54.5% of *conversations*, because one DM redirect anywhere in a thread
makes that whole resolution invisible.

### Problem encountered #13 — booleans printed as 0 and 1

**Symptom:** The `passes_all_gates` column rendered as `0` instead of `False`.

**Root cause:** My table renderer checked `isinstance(value, (int, np.integer))`
before handling booleans. In Python `bool` is a **subclass of `int`**, so `False`
matched the integer branch and was formatted with a thousands separator, giving `0`.

**How I fixed it:** Check `bool` first and render `yes`/`no`. Ordering matters
whenever a type check involves `bool` and `int` together.

### Problem encountered #14 — KeyError on an unanalysed brand

**Symptom:** The full run crashed with `KeyError: 'ATT'`.

**Root cause:** I collect turn signals for the top 25 brands only, but
`brand_metrics` grouped over **all** clean conversations. Brand `ATT` sits outside
the top 25, so it had no entry in the language counters.

**How I fixed it:** Filter the conversations to the analysed brand set before
grouping. Without the filter, brands outside the top 25 would have joined to null
signals and produced meaningless rows — so this was a correctness bug, not just a
crash.

**Note:** the sample harness did **not** catch this one, because the 93-row sample
has only 12 brands and all fall inside the top 25. The full run caught it. The
harness is valuable but not sufficient.

### Gate results

| Gate | Threshold | Brands failing |
| --- | --- | --- |
| G1 Volume | >= 8,000 clean conversations | 0 of the top 25 |
| G2 Public resolution | DM deflection < 40% | 12 |
| G3 Language | >= 95% English | 2 (AmazonHelp 81.9%, AskPlayStation 94.4%) |
| G4 Support purity | < 5% brand-rooted | 1 (Safaricom_Care 6.65%) |

Eleven brands passed all four. Notably eliminated: **AppleSupport** (64.4% DM,
second-largest brand in the dataset), comcastcares (87.8%), TMobileHelp (88.8%),
UPSHelp (74.5%), Tesco (54.5%).

### The score disagreed with the decision, and that is reported

The computed weighted score ranks:

| Rank | Brand | Score |
| --- | --- | --- |
| 1 | GWRHelp | 0.634 |
| 2 | VirginTrains | 0.604 |
| 3 | Delta | 0.602 |
| 6 | **AmericanAir** | **0.569** |

**I did not adjust the weights to make AmericanAir win.** The reason for overriding
the ranking is written in D23: min-max normalisation rewards whichever brand is most
*extreme* on the heaviest criterion, and GWRHelp's 2.6% DM deflection nearly maxes
the 30% public-resolution weight by itself. But GWRHelp is a regional train operator
with 9,577 conversations and an opening vocabulary of `train, paddington, late,
ticket, delayed, cancelled` — roughly four intents and a trivial classifier. The
type-token ratio does not capture "narrow domain" well enough to offset this.

This is a concrete, honest example of a composite metric pointing the wrong way, and
it will be reused in the Phase 17 discussion of misleading headline numbers.

### Selected corpus — AmericanAir

| Metric | Value |
| --- | --- |
| Clean conversations | 24,429 |
| Clean turns | 74,593 (41,460 customer / 33,133 brand) |
| Customer-rooted conversations | 24,239 |
| Retrieval pool | 24,178 |
| Candidate opening messages | 24,239 |
| Multi-turn (4+) | 6,704 |
| Two-turn | 14,809 |
| Truncated | 126 |
| Median turns / duration | 2.0 / 20.1 min |
| DM deflection / URL rate | 21.1% / 8.3% |

### A temporal finding that matters for Phase 6

The report shows "17 months covered", which is misleading in exactly the way this
project keeps running into. The monthly distribution:

| Month | Conversations |
| --- | --- |
| 2014-11 to 2017-09 | 51 combined |
| 2017-10 | 11,854 |
| 2017-11 | 11,440 |
| 2017-12 | 1,084 |

**99.8% of AmericanAir conversations fall in October-December 2017.** A temporal
split will span weeks, not months. Phase 6 must account for this rather than quoting
the 17-month figure.

### Verification

| Check | Result |
| --- | --- |
| Sample harness | Runs; gates evaluate; empty-survivor case handled |
| Clean conversations vs Phase 2 | 24,429 == 24,429 |
| Clean turns vs Phase 2 | 74,593 == 74,593 |
| Customer + brand turns == clean turns | True |
| Truncated count vs Phase 2 | 126 == 126 |
| Determinism | **byte-identical SHA-256 on re-run** |
| Raw data | Unchanged: 516,508,641 bytes, read-only, dated 2019 |
| Phase 1/2 source | Untouched - `git status src/` shows only the new file |

Determinism is clean because no environment-dependent value (free disk, timestamps)
is written into the output.

### Limitations

- Language measurement is a **heuristic**, not a detector. It misses unaccented
  non-English text and may flag English tweets quoting foreign words. Accented
  characters are reported but deliberately excluded from the gate, because English
  tweets routinely contain them.
- Type-token ratio is a crude diversity proxy, measured on a fixed budget of 5,000
  openings per brand so that brands with different volumes stay comparable.
- DM deflection detects the *redirect*, not whether the issue was truly resolved
  privately.
- The top 25 brands cover 71.4% of all clean conversations; smaller brands were not
  analysed.

### Next step

Intent discovery on the AmericanAir corpus: 24,239 customer-rooted opening messages.

---

## Phase 4 — Intent discovery

**Date:** 2026-09-15
**Status:** Complete, with an honest negative result
**Objective:** Find the natural issue structure in AmericanAir opening messages.
Discovery only — no classifier, no retrieval, no golden set.

### What I built

`src/discover_intents.py`: load openings, normalise, TF-IDF, TruncatedSVD, KMeans
sweep, cluster description, quality checks, normalisation-sensitivity check.

### Dependency added

`scikit-learn 1.9.0` (with `scipy 1.17.1`) into the existing `hiver` environment.
Verified the interpreter path is `anaconda3\envs\hiver\python.exe`, so nothing landed
globally. Disk went from 8.7 GB to 7.0 GB free. No torch, no sentence-transformers,
no Ollama, no LLM.

### Count discrepancy, reported rather than hidden

Phase 3 established **24,239** customer-rooted openings. Phase 4 clustered
**24,190**. The difference is **49 messages that normalise to an empty string** —
tweets consisting only of a mention, a URL, or both, for example `@AmericanAir
https://t.co/xyz`. They carry no text to cluster. They are counted and reported, not
silently dropped, and they remain in the corpus for later phases.

### The headline result: the clustering is weak

I did not get a clean intent structure, and the honest reporting of that is the main
output of this phase.

| Measure | Result | Reading |
| --- | --- | --- |
| SVD explained variance (100 components) | **16.56%** | The reduced space keeps little signal |
| Silhouette, every K from 6 to 20 | **0.037 - 0.054** | Essentially no separation |
| Largest cluster at K=20 | **38.3%** | One cluster swallows a third of the corpus |
| Largest cluster at K=6 | 53.5% | Worse at low K |
| ARI between preprocessing variants | **0.37** | Structure is preprocessing-dependent |
| Corpus in plausibly coherent clusters | **37.0%** | Under half lands somewhere meaningful |
| Corpus in `fly`/`flying`/`travel`/`way` clusters | **11.0%** | Split by verb form, not by issue |

Three qualitative failures matter as much as the numbers:

1. **One intent split across clusters.** Delays appear as cluster 5
   (`delayed, flight delayed, hours`) *and* cluster 17 (`hour delay, waiting,
   tarmac, sitting`). Complaints split across clusters 8 and 14. Same meaning,
   different words, different clusters.
2. **A language masquerading as an intent.** Cluster 18 (226 messages, 0.93%) is
   Spanish — `en, que, la, el, por, vuelo`. It has the *highest* cohesion in the run
   (0.871). It is not an intent. Usefully, 0.93% closely matches the 0.8%
   non-English signal the Phase 3 heuristic estimated for AmericanAir, so the two
   independent measurements corroborate each other.
3. **Terms disagreeing with contents.** Cluster 11 holds 15.2% of the corpus and its
   distinctive terms are `cancelled, connecting flight, miss`. But its
   centroid-nearest messages are travel photos, in-flight biscuits and a poll about
   movie watching. The label the terms imply is not what the cluster contains.

**Why this happens:** TF-IDF measures **word overlap**. These messages are short,
informal, emoji-heavy and full of paraphrase. *"bag never showed up"* and *"luggage
missing"* share no tokens at all, so no amount of tuning will place them together.
That is a property of the data, not a parameter problem.

**What I did not do:** tune preprocessing and K until the numbers looked better. The
instruction was explicit, and a tuned partition would still be lexical. Recorded as
D25.

### What did work

The **vocabulary evidence is genuinely informative**, even though the partition is
not. Cluster 2 (1,128 messages, 4.66%) is the standout: terms
`bag, check bag, carry, checked, overhead, space` and the messages agree completely.
Boarding (cluster 1), seats and upgrades (3 and 16) and basic economy (15) are also
coherent. So recurring issue vocabulary is trustworthy; the assignment of every
message to exactly one cluster is not.

### Problem encountered #15 — MKL memory-leak warning on Windows

**Symptom:** Every KMeans call printed `UserWarning: KMeans is known to have a memory
leak on Windows with MKL, when there are less chunks than available threads.`

**Root cause:** A known interaction between scikit-learn's KMeans and Intel MKL when
thread count exceeds the number of data chunks.

**How I fixed it:** `os.environ.setdefault("OMP_NUM_THREADS", "2")` at the very top
of the module, before the sklearn import — it has no effect if set afterwards. This
also helps determinism: thread count changes float summation order, which can perturb
KMeans results slightly.

### Verification

| Check | Result |
| --- | --- |
| Sample harness (500 real openings) | Passed; exercised the full path |
| Messages loaded | 24,190 of 24,239 (49 empty, reported) |
| TF-IDF matrix | 24,190 x 8,126, 99.9% sparse |
| `flight` retained in vocabulary | **yes** - `max_df` kept it, as agreed |
| Agent signatures in customer openings | 2 (effectively absent, as expected) |
| URL-driven clusters | none |
| Format/length-driven clusters | none |

The sample harness used the first 500 real AmericanAir openings rather than the
Phase 2 `_sample` files, because **AmericanAir does not appear in the 93-row sample
at all** — Phase 3 had already reported it as "not present in this source".

### Limitations

- TF-IDF is lexical, not semantic, and cannot match paraphrases.
- KMeans assigns every message to a cluster, including genuine noise, and prefers
  spherical similar-sized groups; real intent distributions are neither.
- Silhouette on sparse text is weak and is reported only for comparison across K.
- No ground truth exists, so nothing is validated.
- Opening messages only; issues raised later in a conversation are invisible.

### Next step

A decision is needed before proceeding: accept a taxonomy authored from term
evidence, or invest in embeddings to get a better partition. The evidence for that
decision is in `reports/phase4_intent_discovery.md` and D25. I have deliberately not
chosen.

---

## Phase 5a/5b — Annotation tooling and candidate taxonomy

**Date:** 2026-09-15
**Status:** Complete. Waiting for human labels before 5c.
**Objective:** Build the sampling tool, guidelines and candidate taxonomy so the
pilot batch can be labelled by hand. **No labels generated.**

### What I built

| File | Purpose |
| --- | --- |
| `src/build_annotation_batch.py` | Draws a stratified sample, writes a blank CSV |
| `golden/taxonomy_v1.md` | Candidate taxonomy, frozen for the pilot only |
| `golden/annotation_guidelines.md` | Labelling rules, edge cases, pre-declared triggers |
| `golden/README.md` | Why this directory is tracked and how labels may be used |
| `golden/b01_pilot_blank.csv` | 148 messages, all label columns empty |

### The sample

Population **24,239** customer-rooted AmericanAir openings — reconciles exactly with
Phase 3, so no definition drifted.

| Stratum | Drawn | Purpose |
| --- | --- | --- |
| `random` | 120 | Uniform, month-proportional. **The only honest frequency estimate** |
| `targeted` | 28 | Keyword-probed so rare intents appear at all. Deliberately unrepresentative |

Month spread of the random stratum: Oct 59, Nov 56, Dec 5 — against a population of
48.9% / 47.2% / 4.5%. Close, as intended.

**28 rather than 30 targeted:** the quota divides evenly across four probes as
7 each. A trivial shortfall, reported rather than padded, since topping it up would
have meant over-drawing from one probe and quietly biasing the stratum.

**Sampling is not based on Phase 4 clusters.** The probes come from *term* evidence,
which held up; cluster membership did not (ARI 0.37). Recorded as D28.

### The guarantee that matters: no automatic labels

This is the point of the phase, so it is enforced in code rather than promised:

- Blank batches are written with **every label column empty**, and there is no code
  path that fills them.
- The writer **refuses any path ending `_labelled.csv`**.
- It **refuses to overwrite an existing blank batch**.
- No LLM is imported or called anywhere in this phase.

**Both guards were tested, not assumed:**

```text
TEST 1  re-run on an existing batch
        -> FileExistsError: ... already exists. Delete it deliberately if you
           intend to redraw; regenerating would change which messages you are
           asked to label.

TEST 2  simulated labelled file present
        -> 24,091 available (148 already labelled, excluded)
```

24,239 - 148 = 24,091, so the exclusion logic that keeps future batches disjoint
works.

### Verification of the blank batch

| Check | Result |
| --- | --- |
| Rows / columns | 148 / 12 |
| **All label columns empty** | **True** (0 non-empty cells of 888) |
| `annotation_id` unique | yes |
| `conversation_id` unique | yes |
| Empty text rows | 0 |
| Embedded newlines surviving into CSV | 0 |
| Text survives the write/read round trip | yes |
| Population vs Phase 3 | 24,239 == 24,239 |

### Practical details that matter more than they look

**`utf-8-sig` encoding.** Plain UTF-8 makes Excel on Windows mangle emoji, and this
corpus is full of them. The BOM fixes it. An annotator fighting mojibake produces
worse labels.

**`QUOTE_ALL`.** Tweets contain commas, quotes and hashes. Unquoted fields would
silently shift columns and corrupt the labels.

**Newlines collapsed to ` / ` in `text_display`.** A tweet with a newline becomes two
rows in a spreadsheet otherwise, which breaks labelling. The original text is
untouched in `conversation_turns.parquet` and joinable on `conversation_id`, so
nothing is lost — this is a rendering, not an edit.

### `OTHER` vs `UNCLEAR`

Kept deliberately separate. `OTHER` means "a real intent the taxonomy lacks" — a gap,
and the most valuable signal the pilot can produce. `UNCLEAR` means "I cannot tell
what they want" — a property of the message. Collapsing them would hide the taxonomy
gap inside message noise.

### Revision triggers, declared before labelling

Fixed in advance so revision cannot be fitted to the results: `OTHER` > 10%; intent
< 2% of the random stratum; `low` confidence > 30% within an intent; primary/secondary
pair > 15%; `UNCLEAR` > 15%. Recorded as D29. **These will not be changed after
seeing the labels.**

### Limitations

- One annotator, so §9's reliability measure is intra-annotator test-retest, **not**
  inter-annotator agreement. Self-agreement is an upper bound on reliability.
- Keyword probes are a lexical bias by construction — hence the stratum flag.
- The taxonomy is a candidate built on weak clustering evidence; the pilot exists
  precisely to test whether it survives contact with real messages.
- Opening messages only; issues that emerge later in a conversation are invisible.

### Next step

**Waiting on human labels.** Nothing further runs until `b01_pilot_labelled.csv`
exists. Stage 5d (analysis) is written only after the labels are returned, so it
cannot be shaped by knowledge of them.

---

## Phase 5d — Pilot annotation analysis

**Date:** 2026-09-15
**Status:** Analysis complete. Taxonomy decision pending joint review.
**Objective:** Determine whether the pilot supports `taxonomy_v1`, using thresholds
declared before any label existed.

### Data integrity — one real problem found

The annotator labelled in Excel and exported to CSV. Two artifacts of that round trip:

**1. `created_at` was reformatted (all 148 rows).** Excel parsed the timestamps and
rewrote them: `2017-10-14 12:13:29` became `14-10-2017 12:13`, **dropping seconds**.
Harmless here — the analysis takes timestamps from
`conversation_turns.parquet`, which is authoritative — but worth knowing, because a
future phase that trusted the CSV timestamps would silently lose precision.

**2. One row's text does not match its `conversation_id`.**

I checked every row's `text_display` against the Phase 2 source rather than assuming
the round trip was lossless. **147 of 148 matched exactly**, emoji and punctuation
included. One did not:

```text
b01_0066  conv_2373654
  CSV shows : "off to @46163 !! / Starting things off right with champagne..."
  source is : "@AmericanAir The largest plane I've ever flown in is the AA 757..."
```

The CSV text appears **nowhere in the blank batch**, so this is not two rows swapping
places — a single cell's content was replaced at some point during editing. The
annotation IDs, conversation IDs and tweet IDs all stayed correctly aligned.

**Handling: excluded from analysis (n=147), not corrected.** The label
(`OTHER`, high confidence, note "couldnot understand") was made against text that
does not belong to that conversation, so it cannot be trusted for `conv_2373654`. I
will not guess which message was actually on screen. It is reported here and in the
generated report, and should be relabelled against the correct source text.

**This check is the reason to compare against source rather than trust a file.**
Nothing else would have caught it: the row looks perfectly well-formed.

### Normalisations — analysis only, file untouched

| Rule | Rows | Change |
| --- | --- | --- |
| `intent_typo` | 3 | `fligh_delay`→`flight_delay`, `baggae`→`baggage`, `booking_fee_and_fare_rules`→`booking_fees_and_fare_rules` |
| `confidence_canonical` | 0 | `med`→`medium` (the annotator used `medium` throughout, so nothing fired) |
| flag values | 105 | `yes` accepted as true |

The guidelines asked for `med` and `y`; the annotator typed `medium` and `yes`, which
are the natural things to type. **That is a flaw in my guidelines, not in the
labelling**, so the analysis accepts both rather than asking for 105 cells to be
retyped. Every affected `annotation_id` is listed in the generated report.

### Results — random stratum only (n=119)

Prevalence uses the random stratum exclusively; the targeted stratum deliberately
over-samples rare intents (D28).

| Intent | n | % | non-high confidence | needs discussion |
| --- | --- | --- | --- | --- |
| flight_delay | 28 | 23.5 | 21% | 31% |
| praise_and_compliment | 14 | 11.8 | 14% | 19% |
| **OTHER** | **14** | **11.8** | 43% | **93%** |
| booking_fees_and_fare_rules | 13 | 10.9 | 31% | 44% |
| baggage | 9 | 7.6 | **0%** | 40% |
| UNCLEAR | 9 | 7.6 | 0% | 55% |
| seating_and_upgrade | 8 | 6.7 | 50% | 50% |
| general_dissatisfaction | 8 | 6.7 | 50% | 56% |
| loyalty_and_lounge | 6 | 5.0 | 50% | 62% |
| boarding_and_gate | 4 | 3.4 | 50% | 40% |
| flight_cancellation_rebooking | 3 | 2.5 | **67%** | **75%** |
| staff_and_service_complaint | 3 | 2.5 | 0% | 25% |

**Triggers fired: 2 of 5.**

- `OTHER` at **11.8%** exceeds the 10% threshold.
- Non-high confidence above 30% in **six** intents:
  `flight_cancellation_rebooking` (67%), `seating_and_upgrade`, `boarding_and_gate`,
  `general_dissatisfaction`, `loyalty_and_lounge` (all 50%),
  `booking_fees_and_fare_rules` (31%).

Not fired: `UNCLEAR` 7.6% (under 15%); no intent below 2%; no primary/secondary pair
above 15%.

### Is `OTHER` one missing category? No — and that matters

The trigger fired, but firing a trigger is not a conclusion. Reading all 14 `OTHER`
rows and their notes, they split into **two groups that are not one category**:

**Group A — non-support social/travel commentary (6 rows).** Photos, views,
news-sharing, affection. *"Always a nice view flying in and out of San Diego"*,
*"How cool! AA Announces New Service to Reykjavik"*, *"AA is still bae"*. No request,
and **not praise of service either** — the annotator consistently chose `OTHER` over
`praise_and_compliment`, which is a meaningful distinction.

**Group B — real support issues, each different (7 rows).** Airport shuttle/ground
transport · accessibility/assistant confusion · broken complaint form (web
technical) · passenger paperwork · security/PreCheck · unreachable phone lines ·
a compensation offer dispute.

**Group B is a long tail, not a category.** Seven issues, seven different topics. That
argues *against* inventing one large catch-all intent and *for* keeping `OTHER` as an
honest residual class.

### `needs_discussion` at 45% — concentrated, not liberal

The rate looked alarmingly high, so I checked whether it was a habit or a signal:

| Intent | % flagged |
| --- | --- |
| OTHER | 93% |
| flight_cancellation_rebooking | 75% |
| loyalty_and_lounge | 62% |
| general_dissatisfaction | 56% |
| flight_delay | 31% |
| staff_and_service_complaint | 25% |
| **praise_and_compliment** | **19%** |

It tracks taxonomy difficulty almost perfectly. The flag is concentrated exactly
where the taxonomy fails and is lowest on the cleanest intent. **This is a
high-quality signal, not over-flagging.**

### `UNCLEAR` is partly absorbing a taxonomy gap

Of the 11 `UNCLEAR` rows, most are genuinely unlabelable (*"Booooooooooo
@AmericanAir 🙄"*, a Spanish message, lounge check-ins). But two are **clear requests
with no home in v1**:

- `b01_0099` — *"WiFi marketed flight but no WiFi available"* — an unambiguous
  in-flight service issue
- `b01_0123` — *"Got married, need to change my last name on my AAdvantage"* — an
  unambiguous account-change request

So the true taxonomy-gap rate is slightly higher than the `OTHER` count alone shows.
**I have not relabelled these** — that is the annotator's call.

### Agreement — not computable, and not invented

**One annotator, one pass.** Neither inter-annotator nor intra-annotator agreement
can be calculated from this pilot, and no score is reported.

A blind test-retest would require the same annotator relabelling a shuffled subset of
40-50 rows after a gap, with the original labels hidden. Cohen's kappa would then be
computable, and would still be an **upper bound** on reliability, since a person
agrees with themselves more than two people agree with each other.

### Sampling check

Manifest verified against the labelled file: seed 42, population 24,239, 120 random +
28 targeted, month split 59/56/5. Matches. No sampling imbalance beyond the
intentional targeted stratum.

### Recommendation — minimal revision (option B), decision pending

Not "freeze as-is": two triggers fired and six intents show weak confidence.
Not "substantially redesign": 10 of 12 labels are working, `baggage` is perfect
(0% non-high confidence), and the distribution is plausible for airline support.

Smallest defensible change, for joint review:

1. **Add `non_support_commentary`** — covers Group A plus several `UNCLEAR` rows.
   Evidence: 6 `OTHER` + ~4 `UNCLEAR`, roughly 8% combined. *Benefit:* removes the
   largest coherent chunk of `OTHER` and gives the agent a clean auto-handle class.
   *Downside:* boundary with `praise_and_compliment` needs a sharp rule.
2. **Add `inflight_experience`** — WiFi, seat comfort, entertainment, catering.
   Evidence: `b01_0099` plus related notes. *Benefit:* a recurring real issue with no
   home. *Downside:* thin pilot evidence; may stay rare.
3. **Merge `flight_cancellation_rebooking` into `flight_delay`** as
   `flight_disruption`, or sharpen its definition. Evidence: 2.5% prevalence, **67%
   non-high confidence, 75% flagged** — the weakest intent, exactly as Phase 4
   predicted when its cluster's terms disagreed with its contents. *Downside:*
   cancellation and delay imply different actions; merging loses that.
4. **Sharpen definitions** for `seating_and_upgrade`, `boarding_and_gate`,
   `general_dissatisfaction`, `loyalty_and_lounge` (all 50% non-high) without
   changing their scope.
5. **Keep `OTHER`** as a residual class. Group B proves a genuine long tail exists.

**Deliberately not proposed:** splitting `seating_and_upgrade` (the Phase 4 open
question). The pilot gives 8 random examples — too few to support a split.

### Limitations

- 119 representative rows; per-intent counts are small and confidence intervals wide.
- One annotator, one pass, no agreement measure.
- Opening messages only.
- `b01_0066` excluded pending relabelling.

### Next step

Joint review of the five proposals, then either freeze v1 or author v2. **No
classifier, retrieval, generation or escalation work until that is settled.**
