import React, { useEffect, useRef, useState } from 'react';
import io from 'socket.io-client';

const CameraStream = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [socket, setSocket] = useState(null);
  const [word, setWord] = useState([]);
  const [answers, setAnswers] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  // Backend URL: use REACT_APP_BACKEND_URL in production (must be HTTPS when frontend is on HTTPS).
  // For local dev, defaults to localhost. When frontend is on HTTPS (e.g. Vercel), backend must use HTTPS so Socket.io uses wss://.
  const BACKEND_URL =
    process.env.REACT_APP_BACKEND_URL ||
    (typeof window !== 'undefined' && window.location.protocol === 'https:'
      ? 'https://18.224.214.141:5000/'
      : 'http://18.224.214.141:5000/');


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

    // Connect to backend using environment variable
    const newSocket = io(BACKEND_URL, {
      transports: ["websocket", "polling"], // Add polling as fallback
      reconnection: true,
      reconnectionAttempts: 10,
      reconnectionDelay: 1000
    });

    newSocket.on('connect', () => {
      console.log('✅ Connected to backend:', BACKEND_URL);
      setSocket(newSocket);
    });

    newSocket.on('connect_error', (error) => {
      console.error('❌ Connection error:', error.message);
    });

    newSocket.on('disconnect', (reason) => {
      console.log('⚠️ Disconnected:', reason);
    });

    newSocket.on('receive_word', (data) => {
      setWord(data.message);
      setAnswers(data.answers);
    });

    return () => {
      if (newSocket) {
        newSocket.disconnect();
      }
    };
  }, []); // Empty dependency array

  useEffect(() => {
    const renderInterval = setInterval(() => {
      renderFrame();
    }, 10);

    const sendInterval = setInterval(() => {
      sendFrame();
    }, 75);

    return () => {
      clearInterval(renderInterval);
      clearInterval(sendInterval);
    };
  }, [socket]);

  const renderFrame = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    const context = canvas.getContext('2d');
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
  };

  const sendFrame = () => {
    const canvas = canvasRef.current;
    const frameData = canvas.toDataURL('image/jpeg');

    if (socket) {
      socket.emit('send_frame', { frame: frameData });
    }
  };

  const skipWord = () => {
    const newAnswers = [...answers];

    if (newAnswers[currentIndex] !== 'correct') {
      newAnswers[currentIndex] = 'skipped';
    }
    setAnswers(newAnswers);

    if (socket) {
      socket.emit('skip_word', { message: word, answers: newAnswers });
    }

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
                    rel="noreferrer"
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