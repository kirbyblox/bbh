# Bridge Bid Hallucinator
Bridge bot that hallucinates bids based on past bids.

## Live Demo

Try it out at: [https://bbh-amber.vercel.app/](https://bbh-amber.vercel.app/)

## Features

- Interactive bidding interface with support for:
  - Standard bid levels (1-7)
  - All suits (♣, ♦, ♥, ♠, NT)
  - Special bids (Pass, Double)
- Neural network-powered bid generation
- Single bid generation mode
- Complete auction generation
- Responsive design with Tailwind CSS

## Stack

- Next.js
- React
- ONNX Runtime Web
- Tailwind CSS

## Usage

### Manual Bidding
1. Select a bid level (1-7)
2. Choose a suit (♣, ♦, ♥, ♠, NT)
3. Use special bids (Pass, X) as needed
4. Click any bid in the sequence to remove it and subsequent bids

### Automated Bidding
- Click "+1" to generate a single bid
- Click "Generate" to complete the entire auction
