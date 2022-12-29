from live_streaming_server import LiveStreamingServer

if __name__ == "__main__":
    server = LiveStreamingServer(debug=True)
    server.start()
