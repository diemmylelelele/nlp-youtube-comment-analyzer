from youtube_summary_tool import analyze_youtube_comments, answer_question, extract_video_id, CURRENT_VIDEO_ID, close_chroma_connection
from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import json
import time
import matplotlib

# Set matplotlib to use a non-interactive backend before any other matplotlib imports
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static')

# Store the latest analysis results
latest_results = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/analyze', methods=['POST'])
def analyze():
    global latest_results

    data = request.json
    youtube_url = data.get('youtube_url')

    if not youtube_url:
        return jsonify({'error': 'No YouTube URL provided'}), 400

    try:
        # Extract video ID to check if we're analyzing a new video
        video_id = extract_video_id(youtube_url)

        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL or video ID'}), 400

        # Make sure any previous connections are closed before analysis
        close_chroma_connection()

        # Run the analysis (the function will reuse the database if it's the same video)
        results = analyze_youtube_comments(youtube_url)

        # Store the results for later use
        latest_results = results

        return jsonify(results)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def ask_question_api():
    global latest_results

    data = request.json
    question = data.get('question')
    k = data.get('k')  # Optional parameter for number of comments to retrieve

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Check if we have a valid database to query
        if not os.path.exists("chroma"):
            if not latest_results:
                return jsonify({'error': 'No analysis has been performed yet. Please analyze a video first.'}), 400
            else:
                video_id = latest_results.get('video_id')
                return jsonify({'error': f'Database not found for video {video_id}. Please re-analyze the video.'}), 400

        # Convert k to integer if it's provided
        if k is not None:
            try:
                k = int(k)
            except ValueError:
                return jsonify({'error': 'Parameter k must be an integer'}), 400

        print(f"Processing question: {question}")

        # Use the improved answer_question function with auto-calculated or user-specified k
        start_time = time.time()
        result = answer_question(question, k=k)

        processing_time = time.time() - start_time
        print(f"Question answered in {processing_time:.2f} seconds")

        # The result already includes the answer, k_used, and other metadata
        return jsonify(result)
    except Exception as e:
        print(f"Error in Q&A: {str(e)}")
        return jsonify({'error': str(e)}), 500


# Add a new endpoint to get database status
@app.route('/api/status', methods=['GET'])
def get_status():
    try:
        status = {
            'database_exists': os.path.exists("chroma"),
            'current_video_id': CURRENT_VIDEO_ID
        }

        # Add metadata from JSON file if it exists
        metadata_path = os.path.join("chroma", "video_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                status['metadata'] = metadata
            except Exception as e:
                status['metadata_error'] = str(e)

        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


# Serve files from root directory
@app.route('/<path:path>')
def serve_root_files(path):
    if os.path.exists(path):
        return send_from_directory('.', path)
    else:
        return "File not found", 404


if __name__ == '__main__':
    app.run(debug=True, port=5000)
