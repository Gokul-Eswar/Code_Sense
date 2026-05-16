# Implementation Plan: End-to-End Optimization

## Objective
Optimize the NLA TUI and Sidecar for seamless, low-latency operations by reducing network overhead and intelligently filtering activation processing.

## Key Constraints & Considerations
- Large activation tensors (e.g., 8192 floats) can bottleneck JSON streaming.
- Processing a thought for every single sub-word token overwhelms the NLA Verbalizer and creates a noisy UX.

## Proposed Solution

### 1. Smart Filtering (Hybrid Approach)
Implement a `ThoughtFilter` that sits between the Interceptor and the Client. It determines if an activation is "worth" translating based on two criteria:
- **Structural Striding**: Always translate if the generated token ends with a space, punctuation (`.`, `,`, `!`, `?`), or newline. This aligns thoughts with human-readable chunks.
- **Semantic Shift (Cosine Similarity)**: For mid-word tokens, calculate the cosine similarity between the current activation vector and the *last translated* activation vector. If the similarity drops below a threshold (e.g., `0.95`, indicating the model's internal representation has significantly shifted), trigger a translation regardless of punctuation.

### 2. Network Payload Compression (Base64)
Modify the `sidecar.py` and `RemoteHTTPProvider` to serialize the `torch.Tensor` (float32) as a Base64 encoded string instead of a JSON array of floats.
- **Sidecar**: `base64.b64encode(tensor.numpy().tobytes()).decode('ascii')`
- **TUI**: `np.frombuffer(base64.b64decode(b64_str), dtype=np.float32)`
- This cuts the payload size by over 50% and dramatically speeds up serialization/deserialization.

### 3. Connection Pooling
Update `NLAClient` to maintain a single, persistent `httpx.AsyncClient` instance across its lifecycle, reusing TCP connections to the SGLang server rather than opening a new connection for every thought.

## Implementation Steps

1.  **Network Optimization**:
    *   Update `sidecar.py` event generator to use Base64 encoding for the `activation` field.
    *   Update `RemoteHTTPProvider` in `nla_tui/providers/remote.py` to decode Base64 back into a `torch.Tensor`.
2.  **Filter Implementation**:
    *   Create `nla_tui/filter.py` with the `ThoughtFilter` class.
3.  **Integration**:
    *   Update `cli.py` and `wrapper.py` to route activations through the `ThoughtFilter` before calling `client.get_thought()`.
    *   Update `client.py` to use a persistent `httpx.AsyncClient`.

## Verification
- Verify the sidecar streams Base64 successfully and the TUI decodes it.
- Observe the "Mind Stream" pane during generation: it should update fluidly at word boundaries or during major concept shifts, rather than flashing wildly for every sub-token.