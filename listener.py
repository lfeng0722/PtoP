import threading
from flask import Flask, jsonify
from cyber.python.cyber_py3 import cyber
from modules.common_msgs.localization_msgs.localization_pb2 import LocalizationEstimate

# Global variable storing the most recent localization data as a dict.
latest_localization = {}

# Previous x, y coordinates used to filter out small-magnitude position updates.
last_x = None
last_y = None

# Application-level Flask instance.
app = Flask(__name__)

# Disable HTTP caching for all responses so clients always receive the latest data.
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

@app.route('/var', methods=['GET'])
def get_variable():
    """
    Return the most recent localization data as JSON.
    Returns an empty dict if no data has been received yet.
    """
    return jsonify(latest_localization)

def run_flask():
    """
    Start the Flask server, listening on 127.0.0.1:5000.
    """
    app.run(host='127.0.0.1', port=5000)

def localization_callback(data):
    """
    Cyber RT callback for localization messages.

    Updates latest_localization only when the ego position has changed by more
    than 5 meters in x or y relative to the last recorded position. This prevents
    flooding the REST endpoint with redundant data on stationary updates.
    """
    global latest_localization, last_x, last_y

    current_x = data.pose.position.x
    current_y = data.pose.position.y

    # First message received: initialize the reference position and publish immediately.
    if last_x is None or last_y is None:
        last_x = current_x
        last_y = current_y
        latest_localization = {
            'timestamp': data.header.timestamp_sec,
            'position': {
                'x': current_x,
                'y': current_y,
                'z': data.pose.position.z
            },
            'orientation': {
                'qw': data.pose.orientation.qw,
                'qx': data.pose.orientation.qx,
                'qy': data.pose.orientation.qy,
                'qz': data.pose.orientation.qz
            }
        }
        print("=" * 80)
        print("Received first localization data (automatically sent):")
        print(f"Timestamp: {data.header.timestamp_sec}")
        print(f"Position: x={current_x}, y={current_y}, z={data.pose.position.z}")
        print(f"Orientation: qw={data.pose.orientation.qw}, qx={data.pose.orientation.qx}, "
              f"qy={data.pose.orientation.qy}, qz={data.pose.orientation.qz}")
        print("=" * 80)
    else:
        # Update only when x or y has changed by more than 5 meters.
        if abs(current_x - last_x) > 5 or abs(current_y - last_y) > 5:
            last_x = current_x
            last_y = current_y

            latest_localization = {
                'timestamp': data.header.timestamp_sec,
                'position': {
                    'x': current_x,
                    'y': current_y,
                    'z': data.pose.position.z
                },
                'orientation': {
                    'qw': data.pose.orientation.qw,
                    'qx': data.pose.orientation.qx,
                    'qy': data.pose.orientation.qy,
                    'qz': data.pose.orientation.qz
                }
            }

            print("=" * 80)
            print("Localization changed by more than 5 (x or y), sending data:")
            print(f"Timestamp: {data.header.timestamp_sec}")
            print(f"Position: x={current_x}, y={current_y}, z={data.pose.position.z}")
            print(f"Orientation: qw={data.pose.orientation.qw}, qx={data.pose.orientation.qx}, "
                  f"qy={data.pose.orientation.qy}, qz={data.pose.orientation.qz}")
            print("=" * 80)
        # If the change is below the threshold, skip the update.

def localization_listener():
    """
    Create a Cyber RT reader for localization messages and spin until shutdown.
    """
    print("=" * 120)
    node = cyber.Node("localization_listener")
    node.create_reader("/apollo/localization/pose", LocalizationEstimate, localization_callback)
    node.spin()

if __name__ == '__main__':
    cyber.init()

    # Start the Flask server in a daemon thread so it exits when the main thread exits.
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.setDaemon(True)
    flask_thread.start()

    # Start listening for localization data.
    localization_listener()
    cyber.shutdown()
