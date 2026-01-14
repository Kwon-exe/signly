import React, { useState, useEffect } from "react";
import Hint from "./Hint";

function TopHalf() {
  const [words, setWords] = useState([
    { text: "Orange", isCorrect: false },
    { text: "Orange", isCorrect: false },
    { text: "Orange", isCorrect: false }
  ]);
  const [currentIndex, setCurrentIndex] = useState(0);

  const markWordAsCorrect = (index) => {
    const newWords = [...words];
    newWords[index].isCorrect = true;
    setWords(newWords);
    moveToNextWord();
  };

  const moveToNextWord = () => {
    if (currentIndex < words.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  

  return (
    <div className="top-half">
      <div className="card">
        <h1>
          {words.map((word, index) => (
            <span
              key={index}
              className={index === currentIndex ? "current-word" : ""}
              style={{ color: word.isCorrect ? "green" : "white" }}
            >
              {word.text}
            </span>
          ))}
        </h1>
      </div>
    </div>
  );
}

export default TopHalf;