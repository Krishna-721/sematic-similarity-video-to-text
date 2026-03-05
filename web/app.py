"""
Video Scene Understanding - Web Interface
Gradio-based web application for video analysis
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import gradio as gr

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import VideoPipeline


# Global pipeline instance
pipeline: Optional[VideoPipeline] = None


def get_pipeline() -> VideoPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        config = Config.load()
        pipeline = VideoPipeline(config)
    return pipeline


def analyze_video(
    video_file: str,
    skip_faces: bool,
    skip_actions: bool,
    generate_narrative: bool,
    speaker_mapping: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str]:
    """
    Main analysis function for Gradio interface
    
    Returns:
        Tuple of (transcript, scene_json, narrative, screenplay, status)
    """
    if video_file is None:
        return "", "", "", "", "Please upload a video file."
    
    try:
        progress(0, desc="Initializing...")
        pipe = get_pipeline()
        
        # Parse speaker mapping
        speaker_names = None
        if speaker_mapping and speaker_mapping.strip():
            try:
                speaker_names = json.loads(speaker_mapping)
            except json.JSONDecodeError:
                pass
        
        # Create temp output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            
            def progress_callback(stage: str, pct: float):
                progress(pct, desc=stage)
            
            # Run pipeline
            result = pipe.process(
                video_path=video_file,
                output_dir=temp_dir,
                speaker_names=speaker_names,
                skip_faces=skip_faces,
                skip_actions=skip_actions,
                generate_text=generate_narrative,
                progress_callback=progress_callback
            )
            
            # Format outputs
            transcript = result.analysis.full_transcript
            scene_json = json.dumps(result.analysis.to_dict(), indent=2)
            narrative = result.generated_narrative
            screenplay = result.generated_screenplay
            
            status = (
                f"✓ Analysis Complete!\n\n"
                f"Duration: {result.analysis.duration:.2f} seconds\n"
                f"Scenes: {len(result.analysis.scenes)}\n"
                f"Speakers: {len(result.analysis.all_speakers)}\n"
                f"Objects: {len(result.analysis.all_objects)}\n"
                f"Processing Time: {result.processing_time:.2f}s"
            )
            
            return transcript, scene_json, narrative, screenplay, status
    
    except Exception as e:
        return "", "", "", "", f"Error: {str(e)}"


def transcribe_only(video_file: str) -> Tuple[str, str]:
    """Quick transcription only"""
    if video_file is None:
        return "", "Please upload a video file."
    
    try:
        from src.video_processor import VideoProcessor
        from src.speech_recognition import SpeechRecognizer
        
        config = Config.load()
        processor = VideoProcessor(config)
        recognizer = SpeechRecognizer(config)
        
        # Extract audio
        audio_path = processor.extract_audio(video_file)
        
        # Transcribe
        result = recognizer.transcribe(audio_path)
        
        # Generate SRT
        srt = recognizer.to_srt(result)
        
        processor.cleanup()
        
        return result.text, srt
    
    except Exception as e:
        return "", f"Error: {str(e)}"


def detect_scenes_only(video_file: str) -> Tuple[str, str]:
    """Quick scene detection only"""
    if video_file is None:
        return "", "Please upload a video file."
    
    try:
        from src.scene_detection import SceneDetector
        
        config = Config.load()
        detector = SceneDetector(config)
        
        result = detector.detect_scenes(video_file)
        
        # Format scenes
        scenes_text = f"Found {result.total_scenes} scenes:\n\n"
        for scene in result.scenes:
            scenes_text += (
                f"Scene {scene.scene_number}: "
                f"{scene.start_time:.2f}s - {scene.end_time:.2f}s "
                f"(duration: {scene.duration:.2f}s)\n"
            )
        
        # JSON output
        scenes_json = json.dumps([{
            'scene': s.scene_number,
            'start': s.start_time,
            'end': s.end_time,
            'duration': s.duration
        } for s in result.scenes], indent=2)
        
        return scenes_text, scenes_json
    
    except Exception as e:
        return "", f"Error: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application"""
    
    with gr.Blocks(
        title="Video Scene Understanding",
        theme=gr.themes.Soft()
    ) as app:
        
        gr.Markdown("""
        # 🎬 Video Scene Understanding
        
        **Multimodal Video Analysis System**
        
        Upload a video to analyze:
        - 🎤 Speech transcription with speaker identification
        - 👤 Face detection and recognition
        - 🎬 Scene change detection
        - 🏞 Object and environment detection
        - 🧠 Action recognition
        - 📝 AI-generated narrative descriptions
        """)
        
        with gr.Tabs():
            # Full Analysis Tab
            with gr.TabItem("🔬 Full Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="Upload Video")
                        
                        with gr.Accordion("Options", open=False):
                            skip_faces_cb = gr.Checkbox(
                                label="Skip Face Detection",
                                value=False,
                                info="Faster processing, no face recognition"
                            )
                            skip_actions_cb = gr.Checkbox(
                                label="Skip Action Recognition",
                                value=False,
                                info="Faster processing, no action detection"
                            )
                            generate_narrative_cb = gr.Checkbox(
                                label="Generate AI Narrative",
                                value=True,
                                info="Use LLM to generate text (requires API key)"
                            )
                            speaker_map_input = gr.Textbox(
                                label="Speaker Name Mapping (JSON)",
                                placeholder='{"SPEAKER_00": "John", "SPEAKER_01": "Jane"}',
                                info="Optional: Map speaker IDs to names"
                            )
                        
                        analyze_btn = gr.Button("🚀 Analyze Video", variant="primary")
                        status_output = gr.Textbox(label="Status", lines=6)
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("📝 Transcript"):
                                transcript_output = gr.Textbox(
                                    label="Transcript",
                                    lines=20,
                                    show_copy_button=True
                                )
                            
                            with gr.TabItem("📊 Scene Data"):
                                scene_json_output = gr.Code(
                                    label="Scene Analysis (JSON)",
                                    language="json",
                                    lines=20
                                )
                            
                            with gr.TabItem("📖 Narrative"):
                                narrative_output = gr.Textbox(
                                    label="Generated Narrative",
                                    lines=20,
                                    show_copy_button=True
                                )
                            
                            with gr.TabItem("🎭 Screenplay"):
                                screenplay_output = gr.Textbox(
                                    label="Screenplay Format",
                                    lines=20,
                                    show_copy_button=True
                                )
                
                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[
                        video_input,
                        skip_faces_cb,
                        skip_actions_cb,
                        generate_narrative_cb,
                        speaker_map_input
                    ],
                    outputs=[
                        transcript_output,
                        scene_json_output,
                        narrative_output,
                        screenplay_output,
                        status_output
                    ]
                )
            
            # Quick Transcription Tab
            with gr.TabItem("🎤 Quick Transcribe"):
                gr.Markdown("**Get a quick transcription without full analysis**")
                
                with gr.Row():
                    with gr.Column():
                        trans_video_input = gr.Video(label="Upload Video")
                        trans_btn = gr.Button("Transcribe", variant="primary")
                    
                    with gr.Column():
                        trans_text_output = gr.Textbox(
                            label="Transcript",
                            lines=15,
                            show_copy_button=True
                        )
                        trans_srt_output = gr.Textbox(
                            label="SRT Subtitles",
                            lines=15,
                            show_copy_button=True
                        )
                
                trans_btn.click(
                    fn=transcribe_only,
                    inputs=[trans_video_input],
                    outputs=[trans_text_output, trans_srt_output]
                )
            
            # Scene Detection Tab
            with gr.TabItem("🎬 Scene Detection"):
                gr.Markdown("**Detect scene changes in your video**")
                
                with gr.Row():
                    with gr.Column():
                        scene_video_input = gr.Video(label="Upload Video")
                        scene_btn = gr.Button("Detect Scenes", variant="primary")
                    
                    with gr.Column():
                        scene_text_output = gr.Textbox(
                            label="Detected Scenes",
                            lines=15
                        )
                        scene_json_out = gr.Code(
                            label="Scene Data (JSON)",
                            language="json",
                            lines=15
                        )
                
                scene_btn.click(
                    fn=detect_scenes_only,
                    inputs=[scene_video_input],
                    outputs=[scene_text_output, scene_json_out]
                )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings"):
                gr.Markdown("""
                ### Configuration
                
                The system uses `config/config.yaml` for settings.
                
                **Required API Keys (set in environment or `.env` file):**
                - `OPENAI_API_KEY` - For GPT narrative generation
                - `HF_TOKEN` - For speaker diarization (pyannote)
                
                **Optional API Keys:**
                - `ANTHROPIC_API_KEY` - For Claude
                - `GOOGLE_API_KEY` - For Gemini
                
                ### Adding Known Faces
                
                To recognize specific people:
                1. Create folder: `known_faces/PersonName/`
                2. Add clear face images (jpg/png)
                3. The system will recognize them in videos
                
                ### Hardware Requirements
                
                - **CPU**: Works on any modern CPU (slower)
                - **GPU**: NVIDIA GPU with 6GB+ VRAM recommended
                - **RAM**: 16GB+ recommended for large videos
                """)
        
        gr.Markdown("""
        ---
        **Video Scene Understanding** | Built with 🎬 OpenCV, 🎤 Whisper, 🔍 YOLO, 🤖 GPT
        """)
    
    return app


def main():
    """Run the web application"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Scene Understanding Web Interface")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    
    args = parser.parse_args()
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
