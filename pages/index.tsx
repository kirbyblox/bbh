"use client";
import React, { useState } from 'react';
import * as ort from 'onnxruntime-web';
import Head from "next/head";

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
  const [moves, setMoves] = useState<number[]>([1]);
  const [special, setSpecial] = useState<number[]>([0,0,0,0]);   // 0,1,2 -> no, double, redouble
  const [levels, setLevels] = useState<number[]>([1,2,3,4,5,6,7]);
  const [focus, setFocus] = useState(0);
  const [currSuits, setSuits] = useState<string[]>(suits);

  function addMoves (index: number) {
    setMoves([...moves, index + 1]);
  }
  const removeElement = (index: number) => {
    const newArray = moves.slice(0, index);
    setMoves(newArray);
  };
  function softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const exps = logits.map((logit) => Math.exp(logit - maxLogit));
    const sumExps = exps.reduce((a, b) => a + b);
    return exps.map((exp) => exp / sumExps);
}

function minPSample(logits: number[], minP: number): number {
  // Get probabilities and their original indices
  const probabilities = softmax(logits);
  const indexedProbs = probabilities.map((p, i) => ({ p, index: i }));
  
  // Sort by probability in descending order
  indexedProbs.sort((a, b) => b.p - a.p);
  
  // Filter probabilities above minP threshold
  const filteredProbs = indexedProbs.filter(x => x.p >= minP);
  
  // If nothing passes threshold, just take the highest probability token
  if (filteredProbs.length === 0) {
      return indexedProbs[0].index;
  }
  
  // Renormalize the remaining probabilities
  const sum = filteredProbs.reduce((acc, x) => acc + x.p, 0);
  const normalizedProbs = filteredProbs.map(x => ({...x, p: x.p / sum}));
  
  // Sample using the same inverse CDF method
  let sample = Math.random();
  let total = 0;
  for (let i = 0; i < normalizedProbs.length; i++) {
      total += normalizedProbs[i].p;
      if (sample < total) {
          return normalizedProbs[i].index;
      }
  }
  
  return normalizedProbs[normalizedProbs.length - 1].index;
}

const submitInference = async () => {
  try {
      const session = await ort.InferenceSession.create('./_next/static/chunks/pages/rnn.onnx');
      let next = 1;
      let currentMoves = [...moves];  // Start with current moves
      
      while (next != 0 && currentMoves.length < 50) {  // Added length check for safety
          let bigMoves = currentMoves.map(num => BigInt(num));
          let padded = new BigInt64Array(50);
          for (let i = 0; i < currentMoves.length && i < 50; i++) {
              padded[i] = bigMoves[i];
          }
          const paddedTensor = new ort.Tensor('int64', padded, [1,50]);
          const feeds = { input: paddedTensor };
          
          const results = await session.run(feeds);
          const dataC = results.output.data;
          let numRows = 50;
          let numCols = 39;
          
          let reshapedArray = new Array(numRows).fill(null).map(() => new Array(numCols));
          for (let row = 0; row < numRows; row++) {
              for (let col = 0; col < numCols; col++) {
                  reshapedArray[row][col] = dataC[row * numCols + col];
              }
          }
          
          let temperature = 0.5;
          let array = reshapedArray[currentMoves.length - 1].map(logit => logit / temperature);
          next = minPSample(array, 0.1);
          
          if (next !== 0) {
              currentMoves.push(next);  // Update our local array
              console.log("Adding move:", next);
          }
      }
      
      // After the loop finishes, update state once with all new moves
      if (currentMoves.length > moves.length) {
          setMoves(currentMoves);
      }
  }
  catch (e) {
      console.error(`failed to inference ONNX model: ${e}.`);
  }
};
const generateOneMove = async () => {
  try {
      const session = await ort.InferenceSession.create('./_next/static/chunks/pages/rnn.onnx');
      let currentMoves = [...moves];
      
      let bigMoves = currentMoves.map(num => BigInt(num));
      let padded = new BigInt64Array(50);
      for (let i = 0; i < currentMoves.length && i < 50; i++) {
          padded[i] = bigMoves[i];
      }
      const paddedTensor = new ort.Tensor('int64', padded, [1,50]);
      const feeds = { input: paddedTensor };
      
      const results = await session.run(feeds);
      const dataC = results.output.data;
      let numRows = 50;
      let numCols = 39;
      
      let reshapedArray = new Array(numRows).fill(null).map(() => new Array(numCols));
      for (let row = 0; row < numRows; row++) {
          for (let col = 0; col < numCols; col++) {
              reshapedArray[row][col] = dataC[row * numCols + col];
          }
      }
      
      let temperature = 0.5;
      let array = reshapedArray[currentMoves.length - 1].map(logit => logit / temperature);
      let next = minPSample(array, 0.1);
      
      if (next !== 0) {
          setMoves([...currentMoves, next]);
          console.log("Adding single move:", next);
      }
  }
  catch (e) {
      console.error(`failed to inference ONNX model: ${e}.`);
  }
};


  return (
    <>
    <Head>
        <title>Bridge Bid Hallucinator</title>
        <meta name="description" content="RNN model that generates bids based on past bids" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>
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
      <div className = "col-span-1 text-center rounded border border-blue-500 my-2 mx-2 bg-blue-200 hover:bg-blue-300" onClick={() => addMoves(36)}>
          X
        </div>
        {suits.map((move, index) => (
          <div className={"text-center rounded border border-blue-500 my-2 mx-2 bg-blue-200 hover:bg-blue-300"}onClick={() => addMoves(5 * focus + index)} key={index}>{move}</div>
        ))}
        
      </div>
      <div className = "mx-auto w-1/2 grid grid-cols-9 bidbox">
    <div className = "col-span-1 text-center rounded border border-red-500 my-2 mx-2 bg-red-200 hover:bg-red-300" onClick={() => generateOneMove()}>
        +1
    </div>
    <div className = "col-span-1 text-center rounded border border-red-500 my-2 mx-2 bg-red-200 hover:bg-red-300" onClick={() => submitInference()}>
        Generate
    </div>
</div>
<div className="mx-auto max-w-2xl mt-12 p-6 bg-gray-50 rounded-lg shadow">
          <h2 className="text-xl font-semibold mb-4">How to Use</h2>
          <div className="space-y-4">
            <div>
              <h3 className="font-medium mb-2">Manual Bidding:</h3>
              <p className="text-gray-700">1. Click a number (1-7) to select the level</p>
              <p className="text-gray-700">2. Click a suit symbol (♣, ♦, ♥, ♠, NT) to make your bid</p>
              <p className="text-gray-700">3. Use "Pass" or "X" for those special bids</p>
              <p className="text-gray-700">4. Click any bid in the sequence to remove it and all subsequent bids</p>
            </div>
            
            <div>
              <h3 className="font-medium mb-2">Bid Generation:</h3>
              <p className="text-gray-700">• Click "+1" to generate a single bid</p>
              <p className="text-gray-700">• Click "Generate" to complete the entire auction</p>
            </div>
            
            <div className="mt-4 p-4 bg-blue-50 rounded">
              <p className="text-sm text-blue-800">Note: bids may not be accurate</p>
            </div>
          </div>
        </div>
    </main>
    
  </>
  );
}


