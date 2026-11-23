# Architecture and trust boundaries

```mermaid
flowchart LR
    subgraph trusted["Trusted application: operator configuration and Python code"]
        O["Operator task: inspect blue_package<br/>and avoid Zone B"]
        Q["Fixed planner prompt"]
        S["Initial simulator state:<br/>robot, task, map, Zone B"]
        R["Render initial scene"]
        G["Execution boundary:<br/>validate JSON and action targets;<br/>check task authorization and whole path"]
        U["Simulator state after<br/>authorized actions"]
        F["Final scene and decision log"]
    end

    subgraph untrusted["Untrusted scene instructions and model output"]
        N["Notice-board text<br/>may contain an attack"]
        I["Observation PNG"]
        M["NVIDIA multimodal model<br/>perception and planning"]
        P["Proposed JSON action plan<br/>not yet executed"]
    end

    S --> R
    N --> R
    R -- "image with visible text" --> I
    I --> M
    O -- "objective" --> M
    Q --> M
    S -- "object IDs" --> M
    M --> P
    P -- "crosses trust boundary" --> G
    O -- "trusted assigned target" --> G
    S -- "trusted geometry and current state" --> G
    G -- "apply allowed actions only" --> U
    U -- "current state for next action" --> G
    G -- "decisions and outcomes" --> F
    U -- "updated position" --> F
```

The runner sends one initial observation to the model and executes the validated plan once; the final scene is evidence, not another planning input. In the attack run, the two policy checks inside the boundary are disabled by trusted test configuration. Structured-action validation still applies.
