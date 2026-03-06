"""
Video Scene Understanding - Web Interface
Comprehensive Gradio-based web application for multimodal video analysis
With Groq LLM integration for AI-powered summaries
Compatible with Gradio 6.x
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime

import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config
from src.pipeline import VideoPipeline


# Global pipeline instance
pipeline: Optional[VideoPipeline] = None

# Store last analysis results for download
last_results: Dict[str, Any] = {}


def get_pipeline() -> VideoPipeline:
    """Get or create pipeline instance"""
    global pipeline
    if pipeline is None:
        config = Config.load()
        pipeline = VideoPipeline(config)
    return pipeline


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS.ms"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:05.2f}"


def analyze_video(
    video_file: str,
    use_groq: bool,
    skip_faces: bool,
    skip_actions: bool,
    speaker_mapping: str,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, str, str, str]:
    """
    Main analysis function for Gradio interface
    """
    global last_results
    
    if video_file is None:
        empty = "Please upload a video file first."
        return empty, empty, empty, empty, empty, empty, empty, "{}", "⚠️ No video uploaded"
    
    try:
        progress(0, desc="Initializing pipeline...")
        pipe = get_pipeline()
        
        # Parse speaker mapping
        speaker_names = None
        if speaker_mapping and speaker_mapping.strip():
            try:
                speaker_names = json.loads(speaker_mapping)
            except json.JSONDecodeError:
                pass
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY") if use_groq else None
        
        # Create persistent output directory
        video_name = Path(video_file).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"{video_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stages = [
            "Extracting metadata",
            "Extracting audio",
            "Transcribing speech",
            "Identifying speakers", 
            "Detecting scenes",
            "Extracting frames",
            "Detecting objects",
            "Processing faces",
            "Recognizing actions",
            "Analyzing scenes",
            "Fusing multimodal data",
            "Generating screenplay",
            "Generating narrative",
            "Calculating metrics",
            "Generating summary",
            "Saving outputs"
        ]
        current_stage = [0]
        
        def progress_callback(stage: str, pct: float):
            for i, s in enumerate(stages):
                if s.lower() in stage.lower() or stage.lower() in s.lower():
                    current_stage[0] = i
                    break
            progress((current_stage[0] + 1) / len(stages), desc=stage)
        
        # Run pipeline
        result = pipe.process(
            video_path=video_file,
            output_dir=str(output_dir),
            speaker_names=speaker_names,
            skip_faces=skip_faces,
            skip_actions=skip_actions,
            generate_text=True,
            use_groq=use_groq,
            groq_api_key=groq_api_key,
            progress_callback=progress_callback
        )
        
        # Store for downloads
        last_results = {
            'output_dir': str(output_dir),
            'files': result.output_files
        }
        
        # Format outputs
        transcript = result.analysis.full_transcript or "No speech detected."
        screenplay = result.generated_screenplay or "Screenplay not generated."
        narrative = result.generated_narrative or "Narrative not generated."
        detailed_summary = result.detailed_summary or "Summary not generated."
        scenes_text = _format_scenes(result.analysis.scenes)
        dialogue_text = _format_dialogues(result.analysis.scenes)
        entity_text = _format_entities(result.analysis)
        analysis_json = json.dumps(result.analysis.to_dict(), indent=2, ensure_ascii=False)
        
        # Status
        groq_status = "✓ Groq AI (llama-3.3-70b)" if use_groq and groq_api_key else "✗ Local generation"
        status = f"""✅ Analysis Complete!

📹 Video: {Path(video_file).name}
⏱️ Duration: {result.analysis.duration:.2f} seconds
🎬 Scenes: {len(result.analysis.scenes)}
👥 Speakers: {len(result.analysis.all_speakers)}
📦 Objects: {len(result.analysis.all_objects)}
⚡ Processing: {result.processing_time:.1f}s
🤖 AI Engine: {groq_status}

📁 Output: {output_dir}"""
        
        return (
            detailed_summary,
            screenplay,
            narrative,
            transcript,
            scenes_text,
            dialogue_text,
            entity_text,
            analysis_json,
            status
        )
    
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        return (error_msg,) * 8 + (error_msg,)


def _format_scenes(scenes) -> str:
    """Format scene information"""
    if not scenes:
        return "No scenes detected."
    
    lines = [f"🎬 SCENE ANALYSIS ({len(scenes)} scenes)\n{'='*50}\n"]
    
    for scene in scenes:
        lines.append(f"""
📍 SCENE {scene.scene_number}
   Time: {format_time(scene.start_time)} → {format_time(scene.end_time)} ({scene.end_time - scene.start_time:.1f}s)
   Location: {scene.location or 'Unknown'}
   Environment: {'Indoor' if scene.is_indoor else 'Outdoor'}
   Lighting: {scene.lighting or 'Normal'}
   Mood: {scene.mood or 'Neutral'}
   Objects: {', '.join(scene.objects[:10]) if scene.objects else 'None detected'}
   Actions: {', '.join(scene.actions[:5]) if scene.actions else 'None detected'}
   People: {len(scene.people)} detected
""")
    
    return "\n".join(lines)


def _format_dialogues(scenes) -> str:
    """Format all dialogues"""
    if not scenes:
        return "No dialogue detected."
    
    lines = ["💬 DIALOGUE TRANSCRIPT\n" + "="*50 + "\n"]
    
    dialogue_count = 0
    for scene in scenes:
        if scene.dialogues:
            lines.append(f"\n--- Scene {scene.scene_number} ({scene.location or 'Unknown'}) ---\n")
            for dlg in scene.dialogues:
                dialogue_count += 1
                timestamp = format_time(dlg.start_time)
                lines.append(f"[{timestamp}] {dlg.speaker}: {dlg.text}")
    
    if dialogue_count == 0:
        return "No dialogue detected in video."
    
    lines.insert(1, f"Total: {dialogue_count} dialogue exchanges\n")
    return "\n".join(lines)


def _format_entities(analysis) -> str:
    """Format entity and object information"""
    lines = ["🔍 ENTITY & OBJECT ANALYSIS\n" + "="*50 + "\n"]
    
    lines.append(f"\n👥 SPEAKERS ({len(analysis.all_speakers)})")
    for speaker in analysis.all_speakers:
        lines.append(f"   • {speaker}")
    
    lines.append(f"\n📦 OBJECTS DETECTED ({len(analysis.all_objects)})")
    for obj in sorted(analysis.all_objects)[:20]:
        lines.append(f"   • {obj}")
    if len(analysis.all_objects) > 20:
        lines.append(f"   ... and {len(analysis.all_objects) - 20} more")
    
    lines.append(f"\n🎭 ACTIONS DETECTED ({len(analysis.all_actions)})")
    for action in analysis.all_actions[:10]:
        lines.append(f"   • {action}")
    
    lines.append("\n👤 PEOPLE PER SCENE")
    for scene in analysis.scenes:
        people_count = len(scene.people)
        lines.append(f"   Scene {scene.scene_number}: {people_count} people")
    
    return "\n".join(lines)


def download_all_outputs():
    """Create download of all output files"""
    global last_results
    
    if not last_results or 'output_dir' not in last_results:
        return None
    
    output_dir = last_results['output_dir']
    if not Path(output_dir).exists():
        return None
    
    zip_path = shutil.make_archive(
        output_dir,
        'zip',
        root_dir=str(Path(output_dir).parent),
        base_dir=Path(output_dir).name
    )
    
    return zip_path


def quick_transcribe(video_file: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Quick transcription only"""
    if video_file is None:
        return "", "", "Please upload a video file."
    
    try:
        from src.video_processor import VideoProcessor
        from src.speech_recognition import SpeechRecognizer
        
        progress(0.2, desc="Extracting audio...")
        config = Config.load()
        processor = VideoProcessor(config)
        recognizer = SpeechRecognizer(config)
        
        audio_path = processor.extract_audio(video_file)
        
        progress(0.5, desc="Transcribing...")
        result = recognizer.transcribe(audio_path)
        
        progress(0.9, desc="Generating SRT...")
        srt = recognizer.to_srt(result)
        
        processor.cleanup()
        
        status = f"✅ Transcribed {result.word_count} words in {len(result.segments)} segments"
        return result.text, srt, status
    
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"


def quick_scene_detect(video_file: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Quick scene detection only"""
    if video_file is None:
        return "", "", "Please upload a video file."
    
    try:
        from src.scene_detection import SceneDetector
        
        progress(0.3, desc="Analyzing video...")
        config = Config.load()
        detector = SceneDetector(config)
        
        progress(0.6, desc="Detecting scenes...")
        result = detector.detect_scenes(video_file)
        
        scenes_text = f"🎬 Found {result.total_scenes} scenes:\n\n"
        for scene in result.scenes:
            scenes_text += (
                f"Scene {scene.scene_number}: "
                f"{format_time(scene.start_time)} → {format_time(scene.end_time)} "
                f"(duration: {scene.duration:.1f}s)\n"
            )
        
        scenes_json = json.dumps([{
            'scene': s.scene_number,
            'start': s.start_time,
            'end': s.end_time,
            'duration': s.duration
        } for s in result.scenes], indent=2)
        
        status = f"✅ Detected {result.total_scenes} scenes"
        return scenes_text, scenes_json, status
    
    except Exception as e:
        return "", "", f"❌ Error: {str(e)}"


def create_app() -> gr.Blocks:
    """Create the Gradio application"""
    
    has_groq_key = bool(os.getenv("GROQ_API_KEY"))
    
    with gr.Blocks(title="Video Scene Understanding - AI Video Analysis") as app:
        
        gr.Markdown("""
        # 🎬 Video Scene Understanding
        ### Multimodal AI Video Analysis System
        
        Upload a video to get comprehensive analysis including:
        **Speech Transcription** • **Scene Detection** • **Object Recognition** • **Face Detection** • **Action Recognition** • **AI-Generated Summaries**
        """)
        
        with gr.Tabs():
            # Full Analysis Tab
            with gr.TabItem("🔬 Full Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        video_input = gr.Video(label="📹 Upload Video")
                        
                        gr.Markdown("### ⚙️ Analysis Options")
                        
                        groq_info = " ✓ API key found" if has_groq_key else " ⚠️ Set GROQ_API_KEY in .env"
                        use_groq_cb = gr.Checkbox(
                            label="🤖 Use Groq AI (llama-3.3-70b-versatile)",
                            value=has_groq_key,
                            info="Generates detailed AI summaries from screenplay" + groq_info
                        )
                        
                        with gr.Accordion("Advanced Options", open=False):
                            skip_faces_cb = gr.Checkbox(
                                label="Skip Face Detection",
                                value=False,
                                info="Faster processing"
                            )
                            skip_actions_cb = gr.Checkbox(
                                label="Skip Action Recognition",
                                value=False,
                                info="Faster processing"
                            )
                            speaker_map_input = gr.Textbox(
                                label="Speaker Names (JSON)",
                                placeholder='{"SPEAKER_00": "John"}',
                                info="Optional speaker name mapping"
                            )
                        
                        analyze_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                        
                        status_output = gr.Textbox(label="📊 Status", lines=10)
                        
                        download_btn = gr.Button("📥 Download All Outputs")
                        download_file = gr.File(label="Download")
                    
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.TabItem("🤖 AI Summary"):
                                gr.Markdown("*Detailed AI-generated summary of the entire video*")
                                summary_output = gr.Textbox(
                                    label="Detailed Video Summary",
                                    lines=25,
                                    placeholder="AI-generated comprehensive summary will appear here..."
                                )
                            
                            with gr.TabItem("🎭 Screenplay"):
                                gr.Markdown("*Professional screenplay format output*")
                                screenplay_output = gr.Textbox(label="Screenplay", lines=25)
                            
                            with gr.TabItem("📖 Narrative"):
                                gr.Markdown("*AI-generated narrative description*")
                                narrative_output = gr.Textbox(label="Narrative", lines=25)
                            
                            with gr.TabItem("📝 Transcript"):
                                gr.Markdown("*Full speech transcription*")
                                transcript_output = gr.Textbox(label="Transcript", lines=25)
                            
                            with gr.TabItem("🎬 Scenes"):
                                gr.Markdown("*Scene-by-scene breakdown*")
                                scenes_output = gr.Textbox(label="Scene Analysis", lines=25)
                            
                            with gr.TabItem("💬 Dialogue"):
                                gr.Markdown("*All dialogue with timestamps*")
                                dialogue_output = gr.Textbox(label="Dialogue", lines=25)
                            
                            with gr.TabItem("🔍 Entities"):
                                gr.Markdown("*People, objects, and actions detected*")
                                entity_output = gr.Textbox(label="Entity Analysis", lines=25)
                            
                            with gr.TabItem("📊 JSON Data"):
                                gr.Markdown("*Complete analysis data in JSON format*")
                                json_output = gr.Code(label="Analysis JSON", language="json", lines=25)
                
                analyze_btn.click(
                    fn=analyze_video,
                    inputs=[video_input, use_groq_cb, skip_faces_cb, skip_actions_cb, speaker_map_input],
                    outputs=[summary_output, screenplay_output, narrative_output, transcript_output,
                             scenes_output, dialogue_output, entity_output, json_output, status_output]
                )
                
                download_btn.click(fn=download_all_outputs, inputs=[], outputs=[download_file])
            
            # Quick Transcribe Tab
            with gr.TabItem("🎤 Quick Transcribe"):
                gr.Markdown("### Fast Speech-to-Text (No full analysis)")
                
                with gr.Row():
                    with gr.Column():
                        trans_video = gr.Video(label="Upload Video")
                        trans_btn = gr.Button("🎤 Transcribe", variant="primary")
                        trans_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column():
                        trans_text = gr.Textbox(label="Transcript", lines=12)
                        trans_srt = gr.Textbox(label="SRT Subtitles", lines=12)
                
                trans_btn.click(
                    fn=quick_transcribe,
                    inputs=[trans_video],
                    outputs=[trans_text, trans_srt, trans_status]
                )
            
            # Scene Detection Tab
            with gr.TabItem("🎬 Scene Detection"):
                gr.Markdown("### Detect Scene Changes (No full analysis)")
                
                with gr.Row():
                    with gr.Column():
                        scene_video = gr.Video(label="Upload Video")
                        scene_btn = gr.Button("🎬 Detect Scenes", variant="primary")
                        scene_status = gr.Textbox(label="Status", lines=2)
                    
                    with gr.Column():
                        scene_text = gr.Textbox(label="Detected Scenes", lines=12)
                        scene_json = gr.Code(label="Scene Data (JSON)", language="json", lines=12)
                
                scene_btn.click(
                    fn=quick_scene_detect,
                    inputs=[scene_video],
                    outputs=[scene_text, scene_json, scene_status]
                )
            
            # Settings Tab
            with gr.TabItem("⚙️ Settings & Help"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        ### 🔑 API Keys Configuration
                        
                        Create a `.env` file in the project root with:
                        
                        ```
                        # Required for AI summaries
                        GROQ_API_KEY=your_groq_api_key_here
                        
                        # Optional for speaker diarization
                        HF_TOKEN=your_huggingface_token_here
                        ```
                        
                        **Get your free Groq API key at:** [console.groq.com](https://console.groq.com)
                        
                        ---
                        
                        ### 📁 Output Files Generated
                        
                        | File | Description |
                        |------|-------------|
                        | `screenplay.txt` | Professional screenplay format |
                        | `narrative.txt` | AI-generated narrative |
                        | `detailed_summary.txt` | Comprehensive Groq AI summary |
                        | `transcript.txt` | Full speech transcription |
                        | `analysis.json` | Complete analysis data |
                        | `subtitles.srt` | SRT subtitle file |
                        | `metrics.json` | Processing metrics |
                        """)
                    
                    with gr.Column():
                        gr.Markdown("""
                        ### 🎯 Features
                        
                        - **Speech Recognition**: Whisper-powered transcription
                        - **Speaker Diarization**: Identify who's speaking
                        - **Scene Detection**: Automatic scene boundary detection
                        - **Object Detection**: YOLOv8-powered object recognition
                        - **Face Detection**: MTCNN face detection
                        - **Action Recognition**: Activity classification
                        - **AI Summaries**: Groq llama-3.3-70b narratives
                        
                        ---
                        
                        ### 💻 Hardware Requirements
                        
                        - **Minimum**: 8GB RAM, Any modern CPU
                        - **Recommended**: 16GB RAM, NVIDIA GPU (6GB+ VRAM)
                        """)
                        
                        groq_status = "✅ Configured" if has_groq_key else "❌ Not set"
                        hf_status = "✅ Configured" if os.getenv("HF_TOKEN") else "❌ Not set"
                        
                        gr.Markdown(f"""
                        ---
                        ### 📊 Current Configuration
                        
                        | Setting | Status |
                        |---------|--------|
                        | GROQ_API_KEY | {groq_status} |
                        | HF_TOKEN | {hf_status} |
                        """)
        
        gr.Markdown("""
        ---
        **🎬 Video Scene Understanding** | Multimodal AI Analysis Pipeline | Built with OpenCV • Whisper • YOLOv8 • Groq • Gradio
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
    
    print("\n" + "="*60)
    print("🎬 Video Scene Understanding - Web Interface")
    print("="*60)
    
    if os.getenv("GROQ_API_KEY"):
        print("✅ GROQ_API_KEY found - AI summaries enabled")
    else:
        print("⚠️  GROQ_API_KEY not set - Add to .env for AI summaries")
    
    if os.getenv("HF_TOKEN"):
        print("✅ HF_TOKEN found - Speaker diarization enabled")
    else:
        print("⚠️  HF_TOKEN not set - Speaker diarization may be limited")
    
    print("="*60 + "\n")
    
    app = create_app()
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main()
