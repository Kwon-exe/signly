import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';  // Import the Socket.IO client

const CameraStream = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);
  const [word, setWord] = useState([]);
  const [answers, setAnswers] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const startVideo = async () => {
      try {
        const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
      }
    };

    startVideo();

    const socket = io("127.0.0.1:5000/", {
        transports: ["websocket"],
        cors: {
          origin: "http://localhost:3000/",
        },
    });
    socket.on('connect', () => {
      console.log('Socket.IO connection established');
      setSocket(socket);
    });

    socket.on('receive_word', (data) => {
        //console.log("word was received", data.message);
        //console.log("answers were received", data.answers)
        setWord(data.message)
        setAnswers(data.answers)
    })

    return () => {
      if (socket) {
        socket.disconnect();
      }
    };
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 75); // 100ms interval

    return () => clearInterval(interval);
  }, [socket]);

  const captureFrame = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert frame to base64 string
    const frameData = canvas.toDataURL('image/jpeg');

    // Send frame via Socket.IO
    if (socket) {
      socket.emit('send_frame', { frame: frameData });
    }
    
  };

  const skipWord = () => {
    const newAnswers = [...answers];

    // Only mark the word as 'skipped' if it's not already 'correct'
    if (newAnswers[currentIndex] !== 'correct') {
        newAnswers[currentIndex] = 'skipped';
    }
    setAnswers(newAnswers);

    // Send the updated sentence and answers back to the server
    if (socket) {
      socket.emit('skip_word', { message: word, answers: newAnswers });
    }

    // Move to the next word
    setCurrentIndex((prevIndex) => (prevIndex + 1) % word.length);
  };

  return (
    <>
      <div className="top-half">
        <div className="card card mb-3 mx-5 mt-5 text-bg-secondary">
          <h1>
            {word.map((wordItem, index) => (
              <div key={index}>
                <span 
                  className={answers[index] === 'correct' ? 'bounce' : ''}
                  style={{
                    color: answers[index] === 'correct' ? 'green' : answers[index] === 'skipped' ? 'grey' : 'black'
                  }}
                >
                  <a 
                    target="_blank" 
                    href={`https://www.signingsavvy.com/sign/${wordItem.toUpperCase()}/2700/1`} 
                    className="text-reset text-decoration-none"
                  >
                    {wordItem}{" "}
                  </a>
                </span>
                
              </div>
            ))}
            <button onClick={() => skipWord()}>Skip</button>
          </h1>
        </div>
      </div>
      <div>
        <video ref={videoRef} autoPlay playsInline width="450" height="450" style={{
            opacity: '0%',
          position: 'absolute',
          top: '500px', 
          left: '800px',
        }}></video>
        <canvas ref={canvasRef} style={{ display: 'none' }} width="640" height="480"></canvas>
      </div>
    </>
  );
};

export default CameraStream;
