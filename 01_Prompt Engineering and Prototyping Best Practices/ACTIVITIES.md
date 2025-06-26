# Assignment for Session 01

## Where to find the Vercel app

https://frontend-nine-blush-80.vercel.app/ 

## A little about the app
I wanted to see how different personas in the prompt affected the result hence this is why the app was built with a "Mood" field in there. One thing bad about the app is that you need to fill in you OpenAI key each time you ask a question.

## Vibe checking
1. Explain the concept of object-oriented programming in simple terms to a complete beginner. 
    - Aspect Tested: I wanted to see how the persona specified in the prompt affected the result. I tested this with the mood of a patient teacher so I added the persona: Very calm and patient with empathy for non IT reader
    - Overall assessment: pretty good. I liked how the LLM was able to use the metaphor of the toys to explain how OOP works. However (and this may be a tall order) you can't really use that metaphor to explain more advanced OOP concepts like inheritence but some may argue that's not "Object" oriented but "Class" oriented. 
    - Response: 
```
Of course! Imagine you have a box of different toys. Each toy has its own features and things it can do. For example, a toy car can move, a doll can talk, and a puzzle piece fits with others. Now, instead of thinking about every single detail separately, we group toys by their types—like all cars, all dolls, all puzzles—and describe what makes each type special.

**Object-oriented programming (OOP)** works kind of like that, but with code. It’s a way to organize your program by creating "objects," which are like the toys. Each object represents a thing (like a car, a person, or a bank account) and has:

1. **Properties** (or attributes) — these are characteristics or details about the object. For example, a car might have a color or speed.
2. **Methods** (or actions) — these are things the object can do. Like the car can start, stop, or honk.

With OOP, you create **classes**, which are like blueprints or templates for objects. The class says what properties and methods an object of that kind will have. Then, you create actual objects based on those blueprints.

This way of programming makes it easier to manage big projects, because you can think about and work with things in your program just like real-world things, in an organized and clear way.

Does that help? Would you like me to give a simple example?
```


2. Read the following paragraph and provide a concise summary of the key points…

    - Aspect Tested: Again I wanted to see how the persona would affect the result so I tested without "Mood" and with Mood: "Very picky and nitpicking English teacher"
    - Overall Assessment: you can see the differences in the wording used in each. Interestingly both have the same word count: 114. The latter seems (subjectively) a little more understandable. 
    - Response:
      - Without Mood: `The paragraph discusses the limitations of traditional dashboards, which often have cluttered interfaces that confuse users, especially non-experts. It highlights the potential of AI, particularly large language models (LLMs), to transform dashboards into interactive, conversational tools that help users find controls, summarize data, and surface important metrics through natural language queries. Emerging solutions like Assistant UI enable AI agents to interact dynamically with dashboard components, creating adaptive interfaces that respond to user intent. The piece also suggests redesigning dashboards for both humans and AI agents, enabling collaborative decision-making and shifting from static observation to proactive diagnosis and action. Ultimately, AI-powered agents can replace rigid alert systems with contextual, flexible insights and recommendations, making dashboards interactive hubs for human-agent collaboration.`
      - With Mood: `The paragraph discusses the limitations of traditional dashboards, which often overwhelm users with complex interfaces and hinder efficient information retrieval, especially for non-experts. It highlights how advanced AI models, particularly large language models (LLMs), can transform dashboards into adaptive, conversational tools that help users find controls, summarize data, and uncover insights through natural language queries. Emerging solutions like Assistant UI enable dynamic, personalized interfaces that adjust based on user intent, potentially replacing static dashboards. The text also envisions dual-mode dashboards tailored for both human users and AI agents, enhancing decision-making and automated responses. Ultimately, dashboards evolve from passive observation platforms into collaborative environments where humans and AI agents jointly analyze and act on data.`
    - Here is the text to be summarized (from section 2 of https://a16z.com/nine-emerging-developer-patterns-for-the-ai-era/): 
```
For years, dashboards have served as the primary interface for interacting with complex systems such as observability stacks, analytics, cloud consoles (think AWS), and more. But their design often suffers from an overloaded UX: too many knobs, charts, and tabs that force users to both hunt for information and figure out how to act on it. Especially for non-power users or across teams, these dashboards can become intimidating or inefficient. Users know what they want to achieve, but not where to look or which filters to apply to get there.

The latest generation of AI models offers a potential shift. Instead of treating dashboards as rigid canvases, we can layer in search and interaction. LLMs can now help users find the right control (“Where can I adjust the rate limiter settings for this API?”); synthesize screen-wide data into digestible insights (“Summarize the error trends across all services in staging over the past 24 hours”); and surface unknown/unknowns (“Given what you know about my business, generate a list of metrics I should pay attention to this quarter”).

We are already seeing technical solutions like Assistant UI that make it possible for agents to leverage React components as tools. Just as content has become dynamic and personalized, UI itself can become adaptive and conversational. A purely static dashboard may soon feel outdated next to a natural language-driven interface that reconfigures based on user intent. For example, instead of clicking through five filters to isolate metrics, a user might say, “Show me anomalies from last weekend in Europe,” and the dashboard reshapes to show that view, complete with summarized trends and relevant logs. Or, even more powerfully, “Why did our NPS score drop last week?”,  and the AI might pull up survey sentiment, correlate it with a product deployment, and generate a short diagnostic narrative.

At a larger scale, if agents are now consumers of software, we may also need to rethink what “dashboards” are or for whom they’re designed. For example, dashboards could render views optimized for agent experience — structured, programmatically accessible surfaces designed to help agents perceive system state, make decisions, and act. This might lead to dual-mode interfaces: one human-facing and one agent-facing, both sharing a common state but tailored to different modes of consumption.

In some ways, agents are stepping into roles once filled by alerts, cron jobs, or condition-based automation, but with far more context and flexibility. Instead of pre-wired logic like if error rate > threshold, send alert, an agent might say, “Error rates are rising. Here’s the likely cause, the impacted services, and a proposed fix.” In this world, dashboards aren’t just places to observe; they’re places where both humans and agents collaborate, synthesize, and take action.
```
3. Write a short, imaginative story (100–150 words) about a robot finding friendship in an unexpected place.
    - Aspect Tested: Since this was a creative piece I wanted to see if the mood affected the result. There are 2 moods: (a) very cheerful (b) Marvin the paranoid android (Hitchiker's Guide reference)
    - Overall assessment: not bad, you can see the mood significantly change with the 2 different personas. 
    - Response:
        - Cheeful mood: `Bleep, the little cleaning robot, had always felt out of place in the bustling city. Every day, it scrubbed floors tirelessly, unnoticed and alone. One rainy afternoon, Bleep’s circuits glitched, and it stumbled into a narrow alley it had never explored. There, amidst forgotten crates, sat a scrappy squirrel named Nibble, shivering and hungry. Surprised by the sudden visitor, Nibble cautiously approached the curious robot. Bleep extended a small compartment filled with spare nuts it had gathered from nearby bins. Nibble’s eyes sparkled as he shared the nuts, chattering happily. From that day on, Bleep and Nibble met daily, swapping stories and sharing tiny treasures. In the hum of the robot’s circuits and the chatter of the squirrel, a unique friendship blossomed—one neither had expected, but both cherished deeply.`
        - Marvin the paranoid Android: `In a quiet scrapyard beneath a fading sun, Rusty, a worn-out robot with flickering circuits, sat alone. Years of neglect had dulled his sensors, and hope was something he only vaguely remembered. One evening, a small, stray cat wandered into the yard, curious about the strange metallic figure. Tentatively, the cat brushed against Rusty’s cold frame. To Rusty’s surprise, his rusted joints hummed softly, a feeling akin to warmth spreading through his circuits. The cat’s gentle purrs became a steady rhythm that Rusty found comforting. Together, they shared silent conversations—Rusty’s blinking lights responding to the cat’s lively eyes. In that forgotten scrapyard, amid twisted metal and silence, Rusty discovered friendship wasn’t about perfect programming or advanced software. It was in the quiet companionship of an unexpected friend, a spark that reignited a long-lost sense of connection.`
4. If a store sells apples in packs of 4 and oranges in packs of 3, how many packs of each do I need to buy to get exactly 12 apples and 9 oranges?
    - Aspect Tested: Since this is a math problem, I added the sentence "Explain how you got the answer" to the question. I tried 2 personas: (a) a 3 year old child (b) a math wizard
    - Overall assessment: The math wizard formatted the answer using LaTeX I believe and was more terse whilst the 3 year old gave a simpler explanation. Both are correct. 
    - Response:
        - 3 year old child: 
```
To get exactly 12 apples and 9 oranges:

- Apples come in packs of 4.
- Oranges come in packs of 3.

**Step 1: Find out how many apple packs you need.**

You want 12 apples and each pack has 4 apples.

Divide 12 by 4:

12 ÷ 4 = 3 packs of apples.

**Step 2: Find out how many orange packs you need.**

You want 9 oranges and each pack has 3 oranges.

Divide 9 by 3:

9 ÷ 3 = 3 packs of oranges.

**Answer:**

You need 3 packs of apples and 3 packs of oranges.

**Explanation:**

Since apples are sold in groups of 4, buying 3 groups gets you exactly 3 × 4 = 12 apples. Similarly, oranges are sold in groups of 3, so buying 3 groups gives you 3 × 3 = 9 oranges. That way, you get exactly the number of fruits you want.
```

        - math wizard 
```
You want to get exactly:

- 12 apples, sold in packs of 4 each
- 9 oranges, sold in packs of 3 each

Let's define variables:

- Let \( A \) = number of apple packs
- Let \( O \) = number of orange packs

Since each apple pack contains 4 apples, the total number of apples is:
\[
4A = 12
\]
Solving for \( A \):
\[
A = \frac{12}{4} = 3
\]

Similarly, each orange pack contains 3 oranges, so the total number of oranges is:
\[
3O = 9
\]
Solving for \( O \):
\[
O = \frac{9}{3} = 3
\]

**Answer:** You need to buy 3 packs of apples and 3 packs of oranges.

**Explanation:** Dividing the desired number of fruits by the number of fruits in each pack gives the number of packs needed. Since both results are whole numbers, you can buy exactly those numbers of packs to get the desired fruits.
```
5. Rewrite the following paragraph in a professional, formal tone…
"For 10 years as a university professor I saw that the only students who were succeeding in a 21st-century workforce were those that were learning new things, using the new things they learned to build awesome stuff, and sharing the awesome stuff they built with others. It wasn’t degrees, prestige, or pedigree; it was making stuff as part of a community that was helping people achieve their dreams."

    - Aspect Tested: (a) No mood and (b) Very nitpicky mood
    - Overall assessment: There are definitely differences in the structure and the language used with the 2 different personas. One interesting thing is that in the one with no mood, Chinese chars were used. 
    - Results:
        - No mood
```
During my ten years as a university professor, I observed that the students who thrived in the contemporary workforce were those actively engaged in continuous learning, applying their newly acquired knowledge to develop innovative projects, and collaboratively sharing their成果 with others. Success was not determined by degrees, prestige, or pedigree, but rather by the ability to create meaningful work within a community dedicated to supporting one another in achieving their goals.
```
        - Very nitpicky mood 
```
During my ten-year tenure as a university professor, I observed that the students who thrived in the 21st-century workforce were those who consistently acquired new knowledge, applied this knowledge to create innovative projects, and collaborated by sharing their accomplishments with others. Success was not determined by degrees, prestige, or pedigree, but rather by actively contributing to a community dedicated to supporting individuals in realizing their aspirations.
```