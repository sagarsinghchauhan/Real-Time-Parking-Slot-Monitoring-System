"""
Streamlit GUI for Smart Parking Detection System
Provides interactive interface for:
- Image/Video selection
- File upload
- Parameter adjustment
- Real-time visualization
"""
# Process every N frames



import streamlit as st
import cv2
import pickle
import cvzone
import numpy as np
import tempfile
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Smart Parking Detection System",
    page_icon="🚗",
    layout="wide"
)

# Title and description
st.title("🚗 Smart Parking Space Detection System")
st.markdown("---")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Mode selection
mode = st.sidebar.radio(
    "Select Input Mode:",
    ["📷 Image", "🎥 Video"]
)

# Parameters
st.sidebar.subheader("Detection Parameters")
width = st.sidebar.number_input("Slot Width (pixels)", min_value=50, max_value=200, value=107)
height = st.sidebar.number_input("Slot Height (pixels)", min_value=30, max_value=100, value=48)
threshold = st.sidebar.slider("Occupancy Threshold", min_value=500, max_value=1000, value=750, step=50)

# Advanced parameters (collapsible)
with st.sidebar.expander("🔧 Advanced Processing Parameters"):
    gaussian_kernel = st.slider("Gaussian Blur Kernel", min_value=3, max_value=9, value=3, step=2)
    adaptive_block = st.slider("Adaptive Threshold Block Size", min_value=11, max_value=51, value=25, step=2)
    adaptive_const = st.slider("Adaptive Threshold Constant", min_value=5, max_value=30, value=16)
    median_kernel = st.slider("Median Blur Kernel", min_value=3, max_value=9, value=5, step=2)
    dilation_iter = st.slider("Dilation Iterations", min_value=1, max_value=5, value=2)

# Display options
st.sidebar.subheader("Display Options")
show_dilated = st.sidebar.checkbox("Show Dilated Image", value=False)
show_pixel_counts = st.sidebar.checkbox("Show Pixel Counts on Slots", value=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** Adjust the threshold if detection is not accurate. Lower values make slots easier to mark as available.")


# Function to check if configuration file exists
def check_config():
    if not os.path.exists('CarParkPos'):
        st.error("⚠️ Configuration file 'CarParkPos' not found!")
        st.info("Please run `carparkpicker.py` first to configure parking slots.")
        return False
    return True


# Function to process image/frame
def process_frame(frame, posList, show_counts=True):
    """Process a single frame and detect parking spaces"""

    # Preprocessing pipeline
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (gaussian_kernel, gaussian_kernel), 1)

    imgThreshold = cv2.adaptiveThreshold(
        img_blur, 250,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adaptive_block, adaptive_const
    )

    imgMedian = cv2.medianBlur(imgThreshold, median_kernel)

    kernel = np.ones((4, 4), np.uint8)
    imgDilate = cv2.dilate(imgMedian, kernel, iterations=dilation_iter)

    # Detection
    spaceCounter = 0
    occupied_count = 0

    for pos in posList:
        x, y = pos

        # Extract ROI
        imgCrop = imgMedian[y:y + height, x:x + width]
        count = cv2.countNonZero(imgCrop)

        # Classification
        if count < threshold:
            color = (0, 255, 0)  # Green - Available
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red - Occupied
            thickness = 2
            occupied_count += 1

        # Draw rectangle
        cv2.rectangle(frame, pos, (pos[0] + width, pos[1] + height), color, thickness)

        # Display pixel count if enabled
        if show_counts:
            cvzone.putTextRect(frame, str(count), (x, y + height - 3),
                               scale=1, thickness=2, offset=0, colorR=color)

    # Display total count
    cvzone.putTextRect(frame, f'Free: {spaceCounter}/{len(posList)}',
                       (50, 50), scale=2, thickness=3, offset=10, colorR=(0, 255, 0))

    return frame, imgDilate, spaceCounter, occupied_count


# Main content area
if mode == "📷 Image":
    st.header("Image Mode")

    # File uploader
    uploaded_file = st.file_uploader("Upload parking lot image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # Check if config exists
        if not check_config():
            st.stop()

        # Load configuration
        with open('CarParkPos', 'rb') as f:
            posList = pickle.load(f)

        st.success(f"✅ Configuration loaded: {len(posList)} parking slots")

        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Process button
        if st.button("🔍 Detect Parking Spaces", type="primary"):
            with st.spinner("Processing image..."):
                # Process image
                processed_img, dilated_img, available, occupied = process_frame(
                    image.copy(), posList, show_pixel_counts
                )

                # Display results
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)

                with col2:
                    st.subheader("Detection Result")
                    st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                # Show dilated image if enabled
                if show_dilated:
                    st.subheader("Dilated Image (Processing Stage)")
                    st.image(dilated_img, use_container_width=True, channels="GRAY")

                # Statistics
                st.markdown("---")
                st.subheader("📊 Statistics")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

                with metric_col1:
                    st.metric("Total Slots", len(posList))
                with metric_col2:
                    st.metric("Available", available, delta=None)
                with metric_col3:
                    st.metric("Occupied", occupied, delta=None)
                with metric_col4:
                    occupancy_rate = (occupied / len(posList)) * 100
                    st.metric("Occupancy Rate", f"{occupancy_rate:.1f}%")

elif mode == "🎥 Video":
    st.header("Video Mode")

    # File uploader
    uploaded_video = st.file_uploader("Upload parking lot video", type=['mp4', 'avi', 'mov', 'mkv'])

    if uploaded_video is not None:
        # Check if config exists
        if not check_config():
            st.stop()

        # Load configuration
        with open('CarParkPos', 'rb') as f:
            posList = pickle.load(f)

        st.success(f"✅ Configuration loaded: {len(posList)} parking slots")

        # Save uploaded video to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name

        # Video processing options
        col1, col2 = st.columns(2)
        with col1:
            process_video = st.button("🎬 Process Video", type="primary")
        with col2:
            frame_skip = st.number_input("Process every N frames", min_value=1, max_value=10, value=2,
                                         help="Skip frames for faster processing")

        if process_video:
            cap = cv2.VideoCapture(video_path)

            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            st.info(f"📹 Video Info: {total_frames} frames, {fps} FPS")

            # Create placeholders
            video_placeholder = st.empty()
            dilated_placeholder = st.empty() if show_dilated else None
            stats_placeholder = st.empty()
            progress_bar = st.progress(0)

            frame_count = 0

            # Process video
            while True:
                ret, frame = cap.read()

                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    break

                frame_count += 1

                # Skip frames for performance
                # if frame_count % frame_skip != 0:
                #     continue

                # Process frame
                processed_frame, dilated_frame, available, occupied = process_frame(
                    frame.copy(), posList, show_pixel_counts
                )

                # Update display
                video_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    caption="Parking Detection",
                    use_container_width=True
                )

                # Show dilated image if enabled
                if show_dilated and dilated_placeholder:
                    dilated_placeholder.image(
                        dilated_frame,
                        caption="Dilated Image (Processing Stage)",
                        use_container_width=True,
                        channels="GRAY"
                    )

                # Update statistics
                occupancy_rate = (occupied / len(posList)) * 100
                stats_placeholder.markdown(f"""
                    ### 📊 Real-time Statistics
                    - **Total Slots:** {len(posList)}
                    - **Available:** {available} 🟢
                    - **Occupied:** {occupied} 🔴
                    - **Occupancy Rate:** {occupancy_rate:.1f}%
                """)

                # Update progress
                progress = frame_count / total_frames
                progress_bar.progress(min(progress, 1.0))

                # Check if user wants to stop
                # if frame_count >= total_frames:
                #     break
                cv2.waitKey(10)
            cap.release()
            st.success("✅ Video processing complete!")

            # Cleanup
            os.unlink(video_path)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p><strong>Smart Parking Detection System</strong> | Powered by Computer Vision</p>
        <p>Built with OpenCV, cvzone, and Streamlit</p>
    </div>
""", unsafe_allow_html=True)

# Instructions in sidebar
with st.sidebar.expander("📖 How to Use"):
    st.markdown("""
    **Step 1:** Run `carparkpicker.py` to configure parking slots

    **Step 2:** Choose input mode (Image or Video)

    **Step 3:** Upload your file

    **Step 4:** Adjust parameters if needed:
    - **Threshold:** Higher = harder to mark as available
    - **Slot Size:** Must match actual parking spaces

    **Step 5:** Click Process to see results!

    **Tips:**
    - Start with default parameters
    - Adjust threshold based on results
    - Enable "Show Dilated Image" for debugging
    """)

with st.sidebar.expander("🎯 Parameter Guide"):
    st.markdown("""
    **Detection Threshold:**
    - < 700: Very sensitive (more false positives)
    - 750: Default (balanced)
    - > 850: Less sensitive (more false negatives)

    **When to Adjust:**
    - Too many false available → Increase threshold
    - Too many false occupied → Decrease threshold

    **Advanced Parameters:**
    Only adjust if default doesn't work well:
    - **Gaussian Blur:** Larger = more smoothing
    - **Block Size:** Larger for bigger lighting variations
    - **Constant:** Higher = less sensitive to changes
    """)