#!/usr/bin/env python3
"""
basic_blockchain.py
A minimal, self-contained blockchain implementation with:
- Block and Blockchain classes
- Proof-of-work mining (simple difficulty target)
- Transaction list per block- Chain validationSave as a file and run with Python 3.7+."""

import time
import json
import hashlib
from typing import List, Dict, Any
def sha256_hex(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()
class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]], previous_hash: str, nonce: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()
    def compute_hash(self) -> str:
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return sha256_hex(block_string)
class Blockchain:
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.current_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self._create_genesis_block()
    def _create_genesis_block(self):
        genesis = Block(0, time.time(), [], "0")
        # For genesis, optionally run proof-of-work to match difficulty
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)
    @property
    def last_block(self) -> Block:
        return self.chain[-1]
    def add_transaction(self, sender: str, recipient: str, amount: float) -> int:
        tx = {"sender": sender, "recipient": recipient, "amount": amount}
        self.current_transactions.append(tx)
        return self.last_block.index + 1
    def proof_of_work(self, block: Block) -> str:
        target_prefix = "0" * self.difficulty
        while not block.hash.startswith(target_prefix):
            block.nonce += 1
            block.hash = block.compute_hash()
        return block.hash
    def mine(self, miner_address: str) -> Block:
        # Reward for mining
        self.add_transaction(sender="NETWORK", recipient=miner_address, amount=1)

        new_block = Block(
            index=self.last_block.index + 1,
            timestamp=time.time(),
            transactions=self.current_transactions.copy(),
            previous_hash=self.last_block.hash,
            nonce=0
        )
        proof = self.proof_of_work(new_block)
        # Clear current transactions and append the mined block
        self.current_transactions = []
        self.chain.append(new_block)
        return new_block
    def is_valid_chain(self, chain: List[Block] = None) -> bool:
        chain = chain or self.chain
        for i in range(1, len(chain)):
            prev = chain[i - 1]
            curr = chain[i]
            if curr.previous_hash != prev.hash:
                return False
            if curr.compute_hash() != curr.hash:
                return False
            if not curr.hash.startswith("0" * self.difficulty):
                return False
        return True
    def to_dict(self) -> List[Dict[str, Any]]:
        return [
            {
                "index": b.index,
                "timestamp": b.timestamp,
                "transactions": b.transactions,
                "previous_hash": b.previous_hash,
                "nonce": b.nonce,
                "hash": b.hash
            } for b in self.chain
        ]
if __name__ == "__main__":
    # Simple demo
    bc = Blockchain(difficulty=4)
    print("Mining block 1...")
    bc.add_transaction("alice", "bob", 10)
    mined = bc.mine(miner_address="miner1")
    print(f"Mined block {mined.index} with hash: {mined.hash}")

    print("Mining block 2...")
    bc.add_transaction("bob", "carol", 5)
    mined = bc.mine(miner_address="miner1")
    print(f"Mined block {mined.index} with hash: {mined.hash}")

    print("\nBlockchain valid?", bc.is_valid_chain())
    print("\nChain:")
    print(json.dumps(bc.to_dict(), indent=2))