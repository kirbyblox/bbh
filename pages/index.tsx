"use client";
import React, { useState } from 'react';
import * as ort from 'onnxruntime-web';

let bridgeMoves: string[] = [];
const suits = ['♣', '♦', '♥', '♠', 'NT'];

// Loop through each level and suit to create the moves
for (let level = 1; level <= 7; level++) {
    for (let suit of suits) {
        bridgeMoves.push(`${level}${suit}`);
    }
}

bridgeMoves.push('Pass', 'X', 'XX');
bridgeMoves.unshift('End');

export default function Home() {
  const [moves, setMoves] = useState<number[]>([1,2,3,4,34]);
  const [special, setSpecial] = useState<number[]>([0,0,0,0]);   // 0,1,2 -> no, double, redouble
  const [levels, setLevels] = useState<number[]>([1,2,3,4,5,6,7]);
  const [focus, setFocus] = useState(1);
  const [currSuits, setSuits] = useState<string[]>(suits);

  function addMoves (index: number) {
    setMoves([...moves, index + 1]);
  }
  const removeElement = (index: number) => {
    const newArray = moves.slice(0, index);
    setMoves(newArray);
  };

  const submitInference = async () => {
    try {
    const session = await ort.InferenceSession.create('./_next/static/chunks/pages/rnn.onnx');
    let bigMoves = moves.map(num => BigInt(num));
    let padded = new BigInt64Array(50);
    // padded[0] = BigInt(0);
    for (let i = 0; i < moves.length && i < 50; i++) {
      padded[i] = bigMoves[i];
    }
    const paddedTensor = new ort.Tensor('int64', padded, [1,50]);
    const feeds = { input: paddedTensor };
    const results = await session.run(feeds);
    const dataC = results.output.data; // should be 50, 39
    let numRows = 50;
    let numCols = 39;

    // Check if the total number of elements matches
    if (dataC.length !== numRows * numCols) {
        throw new Error("The total number of elements does not match the desired shape.");
    }

    // Create the new 2D-like structure
    let reshapedArray = new Array(numRows).fill(null).map(() => new Array(numCols));

    // Copy data into the new structure
    for (let row = 0; row < numRows; row++) {
        for (let col = 0; col < numCols; col++) {
            reshapedArray[row][col] = dataC[row * numCols + col];
        }
    }

    // reshapedArray is now a 2D-like structure of BigInt64Arrays
    let array = reshapedArray[moves.length - 1];
    let indexOfLargest = array.reduce((maxIndex, currentElement, currentIndex, arr) => {
      return currentElement > arr[maxIndex] ? currentIndex : maxIndex;
  }, 0);
    addMoves(indexOfLargest - 1);
    }
    catch (e) {
      console.error(`failed to inference ONNX model: ${e}.`);
    }
  };


  return (
    <main>
      <div className = "mx-auto w-1/3 grid grid-cols-4">
        {moves.map((move, index) => (
          <div className = "text-center rounded border border-blue-500 bg-blue-200 my-2 mx-2 hover:bg-blue-300"  onClick={() => removeElement(index)} key={index}>{bridgeMoves[move]}</div>
        ))}
      </div>
      <div className = "mx-auto w-1/2 grid grid-cols-9 bidbox">
        <div className = "col-span-1 text-center rounded border border-blue-500 my-2 mx-2 bg-blue-200 hover:bg-blue-300" onClick={() => addMoves(35)}>
          Pass
        </div>
        {levels.map((move, index) => (
          <div className={"text-center rounded border border-blue-500 my-2 mx-2 " + (index == focus ? "bg-blue-400" : "bg-blue-200 hover:bg-blue-300")}key={index} onClick={() => setFocus(index)}>{move}</div>
        ))}
      </div>
      <div className = "mx-auto w-1/2 grid grid-cols-9 bidbox">
      <div className = "col-span-1 text-center rounded border border-blue-500 my-2 mx-2 bg-blue-200 hover:bg-blue-300">
          Pass
        </div>
        {suits.map((move, index) => (
          <div className={"text-center rounded border border-blue-500 my-2 mx-2 bg-blue-200 hover:bg-blue-300"}onClick={() => addMoves(5 * focus + index)} key={index}>{move}</div>
        ))}
        
      </div>
      <div className = "mx-auto w-1/2 grid grid-cols-9 bidbox">
        <div className = "col-span-2 text-center rounded border border-red-500 my-2 mx-2 bg-red-200 hover:bg-red-300" onClick={() => submitInference()}>
          Submit
        </div>
      </div>
    </main>
    

  );
}


