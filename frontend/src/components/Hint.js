import React from "react";

function Hint({ toggleImage }) {
  return (
    <div>
    <button className="btn" onClick={toggleImage}>
      <span className="icon">💡</span>
    </button>
    </div>
  );
}

export default Hint;