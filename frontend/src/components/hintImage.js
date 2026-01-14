import React, { useState, useEffect } from "react";
import Hint from "./Hint"; // Import the Hint component

const hints = {
  "TV": "https://www.youtube.com/embed/PQFYm8Nk8o0",
  "smile": "https://www.youtube.com/embed/i-PzqaE5FTY",
  "boat": "https://www.youtube.com/embed/HOmHIzyCO3Q",
  "thankyou": "https://www.youtube.com/embed/EPlhDhll9mw",
  "no": "https://www.youtube.com/embed/yDNcxRM9Nf8",
  "head": "https://www.youtube.com/embed/JoMsp7g1ARc",
  "up": "https://www.youtube.com/embed/uc6uVXnBP2I"
};

function ImageComponent({ show }) {
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (show) {
      fetch("http://127.0.0.1:5000/hint")
        .then(response => response.json())
        .then(data => {
          setUrl(hints[data.data]);  // Update the URL state with the correct hint
        })
        .catch(error => {
          console.error('Error fetching data:', error);
        });
    }
  }, [show]);

  return (
    <>
      {show && url && (
        <div className="video-container">
          <iframe
            width="560"
            height="315"
            src={url}  // Use the dynamically set URL from state
            title="YouTube video player"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          ></iframe>
        </div>
      )}
    </>
  );
}

function App() {
  const [showImage, setShowImage] = useState(false);

  const toggleImage = () => {
    setShowImage(prevState => !prevState); // Toggle the showImage state
  };

  return (
    <div className="hintVideo">
      <Hint toggleImage={toggleImage} />  {/* Render the hint button */}
      <ImageComponent show={showImage} />  {/* Show image if the state is true */}
    </div>
  );
}

export default App;