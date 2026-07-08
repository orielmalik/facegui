# Real-Time Computer Vision & Face Verification Platform

This repository contains a clean, modular Python skeleton architecture for real-time computer vision and face verification. Built around **SOLID** principles, it exposes AI algorithms (OpenCV, MediaPipe, InsightFace) via clean adapter interfaces and presents output streams via a responsive, WebSocket-driven **NiceGUI** dashboard.

---

## 🏗️ Architectural Layout

The project layout decouples application configuration, core infrastructure, design patterns, and independent UI page modules:

```
├── config/
│   ├── config.py                 # Configuration validation via Pydantic
│   └── settings.yaml             # Settings parameters (Camera, Thresholds, Servers)
├── core/
│   ├── exceptions.py             # Infrastructure exception definitions
│   └── logging.py                # Standard Python logging setup
├── patterns/
│   ├── strategy.py               # Similarity calculation Strategy Pattern
│   ├── factory.py                # Vision service Factory Pattern
│   └── observer.py               # Observer / EventHub Pattern
├── services/
│   ├── camera.py                 # CameraManager (OpenCV multi-threaded capture)
│   ├── cv_processor.py           # OpenCVProcessor (color, scale, annotation facade)
│   ├── interfaces.py             # Interface protocols for AI model adapters
│   ├── mediapipe_service.py      # MediaPipe FaceMesh & Pose landmarks adapter
│   ├── insightface_service.py    # InsightFace (ArcFace) embedding adapter
│   ├── similarity.py             # Similarity calculation engine
│   └── face_verification.py      # FaceVerification wrapper (Match/Score scoring)
├── pipeline/
│   └── cv_pipeline.py            # ComputerVisionPipeline (Pipeline Pattern orchestrator)
├── web/
│   ├── dashboard.py              # NiceGUI Dashboard UI layout definition
│   └── websocket_manager.py      # Async WebSocket broker for multi-client broadcasting
├── main.py                       # Main application launcher
├── requirements.txt              # Standard dependencies list
└── .env.example                  # Environment configuration example
```

---

## 🧩 Implemented Design Patterns

### 1. **Facade & Adapter Pattern**
- Complex operations in external libraries are hidden behind simple, cohesive interfaces.
- `CameraManager` wraps OpenCV `VideoCapture`.
- `MediaPipeService` and `InsightFaceService` implement standard interface protocols in [services/interfaces.py](file:///c:/Users/User/PycharmProjects/NiceGuiReco/services/interfaces.py). This allows changing the underlying models (e.g. swap MediaPipe for YOLO Pose) without changing the pipeline or UI.

### 2. **Strategy Pattern**
- Encapsulates comparison logic in [patterns/strategy.py](file:///c:/Users/User/PycharmProjects/NiceGuiReco/patterns/strategy.py).
- Supports runtime switching between `CosineSimilarityStrategy` and `EuclideanSimilarityStrategy`.

### 3. **Factory Pattern**
- Concrete strategy and verification service instantiation is isolated within [patterns/factory.py](file:///c:/Users/User/PycharmProjects/NiceGuiReco/patterns/factory.py), separating creation from run-time execution.

### 4. **Observer / Event Pattern**
- Implemented via [patterns/observer.py](file:///c:/Users/User/PycharmProjects/NiceGuiReco/patterns/observer.py)'s `EventHub`.
- Allows the UI layer and WebSocket broadcasters to subscribe asynchronously (`on_frame_processed`) to events emitted by the background pipeline without coupling the video processor to the web server.

### 5. **Pipeline Pattern**
- Sequentially executes image capture, preprocessing, face/pose tracking, landmark modeling, embedding comparison, and event dispatch in [pipeline/cv_pipeline.py](file:///c:/Users/User/PycharmProjects/NiceGuiReco/pipeline/cv_pipeline.py):
  $$\text{Camera Frame} \to \text{Preprocessing} \to \text{Detection} \to \text{Landmarks} \to \text{Pose} \to \text{Embeddings} \to \text{Similarity} \to \text{Verification} \to \text{UI Update}$$

---

## ⚡ Non-Blocking WebSocket Architecture

NiceGUI runs on an ASGI server (FastAPI/Uvicorn). To keep the video stream responsive and frame rates high, we separate the UI synchronization from the video transmission:
1. **Background Frame Capture:** A background thread in `CameraManager` queries frames continuously, keeping the latest frame in memory.
2. **Background Async Pipeline:** An async loop runs the CV pipeline continuously at the configured frame rate.
3. **Async Broadcasting:** When a frame is ready, the pipeline fires a `frame_processed` event. The async observer translates it to JPEG and broadcasts it over WebSockets.
4. **Client-Side Rendering:** A tiny, highly optimized JavaScript client inside NiceGUI connects directly to `/ws` to draw frames on the screen.

This prevents the NiceGUI Python server from locking up and supports multiple simultaneous browser clients smoothly.

---

## 🚀 Setup & Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Rename `.env.example` to `.env` or set parameters directly inside `config/settings.yaml`.

### 3. Run Application
```bash
python main.py
```
Open `http://localhost:8080` in your web browser.

---

## 📝 Skeleton Extension Guidelines (TODOs)

This project has **no business logic** or database layers. Extend it using the following extension points:
- **Preprocessing:** Customize input resolutions, filters, or rotation matrices in `ComputerVisionPipeline._preprocess()`.
- **Database Integration:** Query registered user profiles or save matches by calling DB libraries inside `FaceVerificationService.verify()`.
- **Anomalies / Logging:** Implement alerts or user blacklist decisions inside `FaceVerificationService.verify()`.
