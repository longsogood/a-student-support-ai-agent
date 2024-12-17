import asyncio
import websockets

async def test_websocket():
    url = "ws://127.0.0.1:8000/chat"
    async with websockets.connect(url) as websocket:
        print("WebSocket connected")

        # Send a message
        await websocket.send("Kể tên một vài câu lạc bộ trường đại học phenikaa")
        # print("Message sent: Hello, server!")

        # Wait for a response
        response = await websocket.recv()
        print(f"Response from server: {response}")

# Run the test
asyncio.run(test_websocket())
