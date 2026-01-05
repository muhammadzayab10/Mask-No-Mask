import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

model = YOLO("best.pt")

# Check what the actual class names are
print("Model class names:", model.names)

def process_image(img):
    """Process single image for mask detection"""
    if img is None:
        return None, "Please upload an image"
    
    try:
        # Convert PIL to BGR for OpenCV
        img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        results = model(img_array, verbose=False)
        
        mask_count = 0
        no_mask_count = 0
        
        # Draw boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            
            # SWAPPED: class 0 is Mask, class 1 is No Mask
            if class_id == 0:  # Changed to 0 for Mask
                color = (0, 255, 0)  # Green in BGR
                label = f"Mask: {confidence:.2f}"
                mask_count += 1
            else:  # class_id == 1 - No Mask
                color = (0, 0, 255)  # Red in BGR
                label = f"No Mask: {confidence:.2f}"
                no_mask_count += 1
            
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
            cv2.putText(img_array, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Convert back to RGB for display
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        # Summary
        total = mask_count + no_mask_count
        if total > 0:
            info = f"👥 Total: {total}\n"
            info += f"😷 Wearing Mask: {mask_count}\n"
            info += f"🚫 No Mask: {no_mask_count}\n"
            if no_mask_count > 0:
                info += f"\n⚠️ Health risk detected!"
            else:
                info += f"\n✅ All safe!"
        else:
            info = "No faces detected"
        
        return img_rgb, info
    
    except Exception as e:
        return None, f"Error: {e}"

def process_video(video_path):
    """Process video for mask detection"""
    if video_path is None:
        return None, "Please upload a video"
    
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return None, "Error: Could not open video file"
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output video path
        output_path = "output_mask_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_mask = 0
        total_no_mask = 0
        violation_frames = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            results = model(frame, verbose=False)
            
            mask_in_frame = 0
            no_mask_in_frame = 0
            
            # Draw boxes
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # SWAPPED: class 0 is Mask, class 1 is No Mask
                if class_id == 0:  # Changed to 0 for Mask
                    color = (0, 255, 0)  # Green in BGR
                    label = f"Mask: {confidence:.2f}"
                    mask_in_frame += 1
                    total_mask += 1
                else:  # class_id == 1 - No Mask
                    color = (0, 0, 255)  # Red in BGR
                    label = f"No Mask: {confidence:.2f}"
                    no_mask_in_frame += 1
                    total_no_mask += 1
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Track violation frames
            if no_mask_in_frame > 0:
                violation_frames += 1
            
            # Add statistics overlay
            status_color = (0, 0, 255) if no_mask_in_frame > 0 else (0, 255, 0)
            cv2.putText(frame, f"Frame: {frame_count+1}/{total_frames}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Mask: {mask_in_frame} | No Mask: {no_mask_in_frame}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            out.write(frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        if not os.path.exists(output_path):
            return None, "Error: Video output file was not created"
        
        # Summary
        info = f"✅ Video processed!\n"
        info += f"📊 Statistics:\n"
        info += f"Total frames: {total_frames}\n"
        info += f"😷 Mask detections: {total_mask}\n"
        info += f"🚫 No mask detections: {total_no_mask}\n"
        info += f"⚠️ Violation frames: {violation_frames}/{total_frames}\n"
        
        if total_no_mask > 0:
            info += f"\n⚠️ Health risk detected!"
        else:
            info += f"\n✅ Full compliance!"
        
        return output_path, info
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing video: {str(e)}\n{traceback.format_exc()}"
        return None, error_msg

# Create Gradio interface with tabs
with gr.Blocks(title="Mask Detection") as demo:
    gr.Markdown("# 😷 Face Mask Detection System")
    gr.Markdown("Detect face masks in images or videos for health safety compliance")
    
    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                image_button = gr.Button("🔍 Detect Masks", variant="primary")
            
            with gr.Column():
                image_output = gr.Image(type="numpy", label="Detection Result")
                image_info = gr.Textbox(label="Health Report", lines=6)
        
        image_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[image_output, image_info]
        )
        
        gr.Markdown("**Legend:** 🟢 Green Box = Wearing Mask | 🔴 Red Box = No Mask")
    
    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                video_button = gr.Button("🎬 Process Video", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="Processed Video")
                video_info = gr.Textbox(label="Processing Report", lines=8)
        
        video_button.click(
            fn=process_video,
            inputs=video_input,
            outputs=[video_output, video_info]
        )
        
        gr.Markdown("**Note:** Video processing may take a few minutes depending on length")

if __name__ == "__main__":
    demo.launch()