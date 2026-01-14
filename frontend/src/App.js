import CameraStream from './components/CameraStream.js';
import React, { useState } from "react";
import "./App.css";
import Navbar from "./components/navbar";
import Tophalf from "./components/TopHalf";
import Hint from "./components/Hint";
import ImageComponent from "./components/hintImage.js";
import HandTrackingWebcam from './components/HandTrackingVideo.jsx';

function App() {
  const [showImage, setShowImage] = useState(false);

  const toggleImage = () => {
    setShowImage((prevShowImage) => !prevShowImage);
  };
  return (
    <div className="App">
      <Navbar />
      <div>
        <CameraStream />
        <Hint toggleImage={toggleImage} />
      </div>

      <ImageComponent show={showImage} />
      <HandTrackingWebcam />
      
    </div>
  );
}

export default App;
