# ROADMAP — Llama-powered XR Agents

This roadmap outlines how to evolve a local Llama model into embodied XR agents that **walk**, **talk**, **perceive** the world, and **act** (navigate, follow, pick/place, idle-wander) across AR/VR/metaverse scenes.

---

## 0) Principles & Targets

**Agent loop:** Perceive → Understand → Plan → Act → Reflect  
**Latency budgets:**  
- Dialogue turn (LLM): ≤ 300 ms streaming first token; ≤ 1.5 s full sentence  
- Control loop (navigation/interaction): 20–60 Hz local; decisions every 100–200 ms  
- Networked multi-user XR: < 120 ms RTT for presence sync

**Core skills:** Spatial reasoning, dialogue, navigation (NavMesh), local avoidance, object affordances, manipulation stubs, social protocols (follow, yield, greet), persistent memory.

---

## 1) Architecture (Foundational)

```

\[XR Client (Unity/Unreal/WebXR)]
├─ Sensors (poses, raycasts, object tags, audio)
├─ Actuators (locomotion, IK, grab, place, emote)
├─ UI (speech bubbles, subtitles)
└─ Agent SDK (Action API + Perception API)
⇅ WebSocket/gRPC
\[Agent Runtime]
├─ Llama Planner (llama.cpp via Python service)
├─ Tooling: Nav/Manip API, World Query, Memory DB
├─ Dialogue Policy (LLM)
├─ Safety/Guardrails
└─ State Manager (belief/world model)
⇅
\[Services]
├─ Vector Memory (local DB)
├─ ASR/TTS (on-device or edge)
└─ Persistence (profiles, scenes, tasks)

````

**Stack choices:**
- **Engine:** Unity (NavMesh, ML-Agents optional) or Unreal (MassAI, Chaos) or WebXR (Three.js/Babylon.js + NavMesh + IK)  
- **Runtime:** Python (Flask/FastAPI) + `llama-cpp-python` (matches app already)  
- **Transport:** WebSocket for streaming actions & chat; gRPC for low-latency binary (later)  
- **Memory:** sqlite + FAISS or Chroma; key-value for working memory

---

## 2) Action & Perception Contracts (MVP)

**Perception (client → agent):**
```json
{
  "agent_id": "npc_01",
  "time": 123.45,
  "self": {"pos":[x,y,z], "rot":[qx,qy,qz,qw], "hand_free": true},
  "targets": [{"id":"cube_12","pos":[...],"tag":"pickup","grabbable":true}],
  "humans": [{"id":"user_1","pos":[...], "gaze":"agent"}],
  "nav": {"reachable": true, "waypoints":[...]},
  "events": ["user_said: can you follow me?"]
}
````

**Actions (agent → client):**

```json
{
  "agent_id":"npc_01",
  "actions":[
    {"type":"Speak", "text":"Sure, I will follow you."},
    {"type":"Follow", "target":"user_1"},
    {"type":"MoveTo", "pos":[x,y,z]},
    {"type":"Pick", "object":"cube_12"},
    {"type":"Place", "object":"cube_12", "pos":[x2,y2,z2]},
    {"type":"Wander", "radius": 3.0, "seconds": 20}
  ]
}
```

**Minimal action set:** `Speak`, `MoveTo`, `LookAt`, `Follow`, `Stop`, `Wander`, `Pick`, `Place`, `Emote`.

---

## 3) Phased Delivery

### Phase 1 — **Talking Walker (2–3 weeks)**

* Integrate existing Flask `/assistant` as **Planner/Dialogue**.
* Unity/Unreal/WebXR client:

  * NavMesh agent + local avoidance (RVO or engine default).
  * `MoveTo`, `Follow`, `Wander` implemented.
  * Speech bubbles; optional on-device TTS.
* World summary payload → Llama prompt template.
* Safety: speed caps, boundary zones, action rate limiting.

**Milestones**

* NPC follows verbal command “Follow me.”
* NPC wanders idly when idle.
* First-token streaming speech.

### Phase 2 — **Object Interaction (3–4 weeks)**

* Add object affordances: tags (`pickup`, `placeable`, `heavy`).
* IK or physics handles for grab/place; stub grasp for VR controllers.
* Llama tool-use head: generate **action JSON** deterministically (`temperature≈0.2`) for actuation; use natural language for chat.
* Memory store: remember user preference (“call me Alex”), last object placed.

**Milestones**

* “Pick the red cube and put it on the table.”
* Memory recall across session.

### Phase 3 — **Spatial Awareness & Tasks (4–6 weeks)**

* Semantic map: regions (`kitchen`, `hall`, `workbench`) + landmarks.
* Task graphs (BT/GOAP): Llama produces **plan steps**, executor refines.
* Multi-modal stubs: image/tag snapshots from scene → captions (optional VLM later).
* Basic interruptions & replanning (“stop”, “actually place it by the window”).

**Milestones**

* “Go to the kitchen, grab a mug, bring it here.”
* Robust interrupt / resume.

### Phase 4 — **Multi-Agent & Social (4–6 weeks)**

* Multiple NPCs; shared blackboard; simple role assignments.
* Social rules: yield in narrow spaces, group follow, greeting.
* Voice pipeline: on-device ASR/TTS or edge with VAD.

**Milestones**

* Two agents coordinate to move objects.
* Group follow of a user.

### Phase 5 — **Learning & Personalization (6–10 weeks)**

* Episodic memory → vector DB with TTL & pinning.
* Few-shot adapters/prompts per scene or user.
* Optional IL/RL in sim (Unity ML-Agents or Unreal MassAI) for grasp refinements.

**Milestones**

* Agent improves placement accuracy over sessions.
* Personalized behaviors (“walk on my left side”).

---

## 4) Prompting & Policies

**Planner prompt (tool-use style):**

```
System: You are an XR agent. Output ONLY a JSON with an array "actions".
Tools:
- Speak(text)
- MoveTo(pos[x,y,z])
- Follow(target_id)
- Pick(object_id)
- Place(object_id, pos[x,y,z])
- Wander(radius, seconds)

World:
{{WORLD_JSON_SNIPPET}}

User: {{USER_UTTERANCE}}

Rules:
- Keep responses short.
- Prefer Follow for escort-like commands.
- Validate object ids from world.
- Stay within navmesh.

Return:
{"actions":[ ... ]}
```

**Dialogue prompt:** separate, higher temperature, streamed to UI.

---

## 5) Engine Implementations

**Unity**

* `NavMeshAgent` for pathing; `Animator` blend tree for walk/idle/run.
* `XR Interaction Toolkit` for pick/place or custom grabbables.
* WebSocket client to Agent Runtime; JSON actions → coroutines.
* Optional motion matching (Unity MxM) later for natural locomotion.

**Unreal**

* `CharacterMovement` + `NavMeshBoundsVolume`.
* `MassAI` for crowds; `Motion Warping` for alignment; `IK Retargeter`.
* `HTTP/WS` plugin or gRPC for action streaming.

**WebXR**

* Three.js/Babylon.js; recast-navigation / patroljs for NavMesh.
* WebAudio for TTS; WebSocket to planner.

---

## 6) Safety & Sandbox

* Spatial constraints: no-go volumes; speed limits; step height limits.
* Collision budget: max pushes per minute; emergency stop on contact.
* Action watchdog: server validates every action against world.
* Privacy: on-device inference by default; redact PII in transcripts.

---

## 7) Evaluation

**Metrics**

* Task success rate (TSR) per task template.
* Path optimality vs. A\* baseline.
* Grasp/place success (within tolerance).
* Dialogue coherence (human eval / rubric).
* Latency (P50/P95) for plan + act loop.

**Test sets**

* Command paraphrases.
* Cluttered scenes; moving humans; occluded objects.
* Long-horizon tasks with 3–5 steps.

---

## 8) Ops & Packaging

* Run Llama with `llama.cpp` via `llama-cpp-python` (Metal/CUDA).
* **Gunicorn**: `-w 1 -k gthread --threads 8` to keep one model in RAM.
* Config via env: `LLAMA_N_CTX`, `LLAMA_N_GPU_LAYERS`, model path vars.
* Scene service: small service exposing object catalog & region map.

---

## 9) Stretch Goals

* Vision: monocular 3D grounding; on-device VLM (clip-like).
* Motion synthesis: diffusion for gestures/emotes.
* Hierarchical memory: episodic → semantic → identity/persona.
* Tool APIs: doors, elevators, levers; crafting; inventory systems.
* Multi-world metaverse hopping with persistent agent identity.

---

## 10) Minimal Slices (Code Sketches)

**Unity C# — action apply loop (pseudo)**

```csharp
void ApplyAction(Action a) {
  switch (a.type) {
    case "MoveTo": agent.SetDestination(Vec3(a.pos)); break;
    case "Follow": followTarget = Find(a.target); break;
    case "Stop": agent.ResetPath(); followTarget=null; break;
    case "Pick": StartCoroutine(PickRoutine(a.object)); break;
    case "Place": StartCoroutine(PlaceRoutine(a.object, Vec3(a.pos))); break;
    case "Wander": Wander(a.radius, a.seconds); break;
    case "Speak": SubtitleUI.Show(a.text); TTS.Say(a.text); break;
  }
}
```

**Python — planner call (matches existing Flask app)**

```python
payload = {
  "messages":[
    {"role":"system","content":SYSTEM_HINT},
    {"role":"user","content":world_json + "\nUser: " + user_utterance}
  ],
  "max_tokens": 256, "temperature": 0.2, "top_p": 0.9
}
# POST /assistant → stream JSON "actions"
```

---

## Timeline (indicative)

* Weeks 1–3: Phase 1 (Talking Walker)
* Weeks 4–7: Phase 2 (Pick/Place + Memory v1)
* Weeks 8–12: Phase 3 (Tasks & Spatial Semantics)
* Weeks 13–18: Phase 4 (Multi-Agent + Voice)
* Weeks 19–28: Phase 5 (Learning/Personalization)

---

## Risks & Mitigations

* **LLM hallucination → invalid actions**: strict JSON schema, server validation, fallback BT.
* **Latency spikes**: local inference, thread pool, action caching.
* **Physics instability**: snap-to, position tolerances, slow-in/out.
* **User comfort (VR)**: locomotion smoothing, safety bubbles, fade-throughs.

---

## Deliverables

* Agent Runtime (Python) with `/plan` and `/dialogue` endpoints.
* Engine SDKs (Unity/Unreal/WebXR) with Action/Perception adapters.
* Sample scenes, task packs, and eval harness.
* Docs: action schema, prompts, tuning guides.

---

