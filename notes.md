## AI Usage


### Overall design

My overall design was quite simple: I added basic exploration tools for the agent, and then basic knowledge base information. I based everything on a mix of a) my use of claude code, and b) my own personal use of databricks.

I basically framed my work based into the following two steps.

### Step 1: Basic tools

I started by adding 3 tool:, explore schema (list schemas in duckdb), explore table (explore a particular table) and run_query (run a query without submitting it as an answer). 

In this step, I saw two major errors appear.

First, I would frequently see gpt-oss generate invalid sql or function calls like so: 
```json
[
  {
    "id": "chatcmpl-tool-966fe2d8785f205c",
    "type": "function",
    "function": {
      "name": "explore_table",
      "arguments": "{\n  \"table_name\": \"financial.account\",\n  \"limit\": 5\n"   ← no closing brace
    }
  },
  {
    "id": "",
    "type": "function",
    "function": {
      "name": "",            ← phantom: empty name
      "arguments": "}"        ← the missing brace leaked into a second call
    }
  }
]
```
To deal with improper sql, I added a json post-processing function. In step 2, I ended up using pydantic which was a lot cleaner

Second, I regularly saw my model exceed it's "max iteration" budget of 30. It was overly focused on exploring. In response, I did the following. A)  To deal with budget exceeding, I prepended the {current_iter} / {max_iter} to each model response, and explicitly wrote how many iterations the model had left. B) I also ammended the system prompt to remind the model that it should focus more on exploring early in its iteration cycle and submission later.

I saw approximately the final results at the end of step 1. In particular, I was quite shocked because I hadn't added a knowledge base.

```markdown
For reference, current standing:
- Easy: ~73% (47/64), bounded by SQL correctness variance.
- Hard: ~30% (19/64), bounded by missing access to the business-rule guides.
``` 

### Step 2: Adding a basic knowledge base

Next, I added the knowledge base compoennt. Interestingly, when I added these, I actually saw performance drop quite a bit. I initially went from a 70% -> a 38% pass rate on the easy problems, before later recovering to the final numbers below.

First, I implementd a basic knowledge base. I basically just added two tools: explore_guides (list all the guides, with a 'grep' parameter), open_guide (let the agent read a guide). At the end of this first iteration our pass rate actually went down from what it was before. 
  We got a roughly 35% passrate on easies! This told me we needed some more tweaking.

Second, I added some subtle tweaking.I started using pydantic's partial json parsing (https://pydantic.dev/docs/validation/latest/concepts/json/#partial-json-parsing) rather than manually parsing everything which actually substantially boosted performance a lot. I updated the system prompt, and gave it a really really short context on what the knowledge base actually was. Finally, I renamed and tweaked the initial tools. list_guides, open_guide (removed the "after" parameter). In particular, I renamed from explore_guides -> list_guides because it felt more natural to me.
  At the end of this second iteration, we had slightly improved from the end of step 1, we saw the following results

At the end of this step, I did 3 runs on easy/hard to verify how I was doing. I noticed a ton of max iteration errors that really concerned me, and some non-determinism. See my claude code output.
```
┌──────┬────────────┬──────────────┬──────────────┬────────────────────┬────────┐
│ Run  │    Easy    │     Hard     │    Total     │ "Other" (max-iter) │ Tokens │
├──────┼────────────┼──────────────┼──────────────┼────────────────────┼────────┤
│ 1    │ 44 (68.8%) │ 22 (34.4%)   │ 66 (51.6%)   │ 22                 │ 11.3M  │
├──────┼────────────┼──────────────┼──────────────┼────────────────────┼────────┤
│ 2    │ 44 (68.8%) │ 27 (42.2%)   │ 71 (55.5%)   │ 8                  │ 7.0M   │
├──────┼────────────┼──────────────┼──────────────┼────────────────────┼────────┤
│ 3    │ 50 (78.1%) │ 28 (43.8%)   │ 78 (60.9%)   │ 8                  │ 7.95M  │
├──────┼────────────┼──────────────┼──────────────┼────────────────────┼────────┤
│ mean │ 46 (71.9%) │ 25.7 (40.1%) │ 71.7 (56.0%) │ 12.7               │ —      │
└──────┴────────────┴──────────────┴──────────────┴────────────────────┴────────┘
```

Third, I noticed we were still getting a lot of AGENT_ERROR calls. All of my errors except for 1 were "AGENT_ERROR" calls. Turns out, the gpt-oss parsing issue was still not fully solved. Sometimes we would parse a single call into two separate calls, so there was one call and a phantom call. I fixed this issue by parsing the entire original call with pydantic. 

At the end of this step, I noticed significantly less non-determinism. I also noticed fewer token usage (therefore less wandering), and less AGENT_ERROR's. So even though it was roughly a 4% bump, I felt significantly better about our results at this point. Here some claude code output
```
┌──────┬────────────┬──────────────┬──────────────┬─────────────────────────┬────────┐
│ Run  │    Easy    │     Hard     │    Total     │ AGENT_ERROR (E+H, all   │ Tokens │
│      │            │              │              │ max-iter)               │        │
├──────┼────────────┼──────────────┼──────────────┼─────────────────────────┼────────┤
│ 1    │ 48 (75.0%) │ 30 (46.9%)   │ 78 (60.9%)   │ 7   (4 + 3)             │ 6.93M  │
├──────┼────────────┼──────────────┼──────────────┼─────────────────────────┼────────┤
│ 2    │ 46 (71.9%) │ 25 (39.1%)   │ 71 (55.5%)   │ 7   (4 + 3)             │ 6.76M  │
├──────┼────────────┼──────────────┼──────────────┼─────────────────────────┼────────┤
│ 3    │ 51 (79.7%) │ 25 (39.1%)   │ 76 (59.4%)   │ 6   (4 + 2)             │ 6.53M  │
├──────┼────────────┼──────────────┼──────────────┼─────────────────────────┼────────┤
│ mean │ 48.3(75.5%)│ 26.7 (41.7%) │ 75.0 (58.6%) │ 6.7 (4.0 + 2.7)         │ 6.74M  │
└──────┴────────────┴──────────────┴──────────────┴─────────────────────────┴────────┘

Where the remaining gap is: almost entirely MISMATCH — wrong-but-submitted SQL, concentrated on hard (36–37/run). That's the domain-correctness frontier (reading/applying the right guide), not a parsing or infra problem anymore.
```

Andrew (my recruiter) told me to aim for 70% pass rate on easies, and 30% pass rate on the hards. I figured since I was beating that, and was at roughly the ~4 hour marker that I should call it quits. Fixing the sql mismatches is significantly harder.

### What I would do if I had more time

Three things come to mind:

First, I would experiment with using bash commands rather than standard sql . Anecdotally, agents are really good at using bash (https://www.reddit.com/r/AI_Agents/comments/1i2olbq/using_bash_scripting_to_get_ai_agents_make/).

Second, gpt-oss is a solid model but there are better options out there. I'd also explore iterating on the model itself. But for now, I didn't want to waste money on the prompting.

Third, I would want to investigate the wording issues. I think we likely need some sort of query planning or query understanding step. This is probably the next big lever unlock given more time.

## How I used AI

I used Claude code to write all of my code, my initial tool calls were all written in plan mode. I did subsequent iteration instandard mode.

I usually include something in my prompt to tell claude to "grill" me: https://www.aihero.dev/my-grill-me-skill-has-gone-viral. So it's extremely aggressive in the questioning

I probably spent about 4 hours on this assignment, and I probably used $30-40 of claude-opus tokens? I spent another 30 minutes cleaning up my writeup.
