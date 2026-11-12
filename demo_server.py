#!/usr/bin/env python3
"""
Small local web demo for a trained nanoGPT Shakespeare character model.
"""

import argparse
import json
import os
import pickle
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from model import GPT, GPTConfig


ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = ROOT / "out-shakespeare-char-last-good"


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nanoDiffusionGPT - Character Marathon</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #ffffff;
      --paper: #ffffff;
      --bg-2: #f4f4f5;
      --ink: #18181b;
      --ink-2: #3f3f46;
      --ink-3: #71717a;
      --rule: #e4e4e7;
      --rule-2: #f4f4f5;
      --accent: #4caf50;
      --accent-bg: #e8f5e9;
      --warn: #b45309;
      --error: #b12b35;
      --nav-bg: rgba(255, 255, 255, 0.86);
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      --mono: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
    }

    :root[data-theme="dark"] {
      color-scheme: dark;
      --bg: #121312;
      --paper: #151713;
      --bg-2: #1b1f1b;
      --ink: #f4f4f5;
      --ink-2: #d4d4d8;
      --ink-3: #a1a1aa;
      --rule: #303530;
      --rule-2: #232823;
      --accent: #6bcf70;
      --accent-bg: #18351f;
      --warn: #f59e0b;
      --error: #ff6b7a;
      --nav-bg: rgba(18, 19, 18, 0.88);
    }

    * {
      box-sizing: border-box;
    }

    html {
      scroll-behavior: smooth;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 15px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
      text-rendering: optimizelegibility;
    }

    button,
    input,
    textarea {
      font: inherit;
    }

    a {
      color: var(--ink);
      text-decoration-color: var(--rule);
      text-underline-offset: 3px;
    }

    a:hover {
      color: var(--accent);
      text-decoration-color: var(--accent);
    }

    .topnav {
      position: sticky;
      top: 0;
      z-index: 50;
      border-bottom: 1px solid var(--rule);
      background: var(--nav-bg);
      backdrop-filter: blur(8px);
    }

    .topnav-inner {
      display: flex;
      align-items: center;
      gap: 16px;
      max-width: 1080px;
      margin: 0 auto;
      padding: 12px 32px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 9px;
      color: var(--ink);
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
      text-decoration: none;
    }

    .brand-mark {
      position: relative;
      width: 26px;
      height: 26px;
      border: 1px solid var(--ink);
      background:
        linear-gradient(90deg, transparent 44%, var(--accent) 44% 56%, transparent 56%),
        linear-gradient(0deg, transparent 44%, var(--accent) 44% 56%, transparent 56%);
    }

    .brand-mark:after {
      content: "";
      position: absolute;
      width: 8px;
      height: 8px;
      right: 3px;
      bottom: 3px;
      background: var(--ink);
    }

    .brand:hover .brand-mark {
      border-color: var(--accent);
    }

    .nav-links {
      display: flex;
      align-items: center;
      gap: 3px;
      margin-left: auto;
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
    }

    .nav-links a {
      display: inline-flex;
      align-items: center;
      min-height: 34px;
      padding: 0 10px;
      color: var(--ink);
      text-decoration: none;
    }

    .nav-links a:hover,
    .nav-links a:focus-visible {
      color: var(--bg);
      background: var(--ink);
      outline: 0;
    }

    .theme-toggle {
      width: 34px;
      height: 34px;
      border: 1px solid var(--rule);
      background: var(--bg);
      color: var(--ink-2);
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 800;
      cursor: pointer;
    }

    .theme-toggle:hover,
    .theme-toggle:focus-visible {
      color: var(--bg);
      background: var(--ink);
      border-color: var(--ink);
      outline: 0;
    }

    .container {
      max-width: 920px;
      margin: 0 auto;
      padding: 0 32px;
    }

    .wide {
      max-width: 1060px;
    }

    .hero {
      border-bottom: 1px solid var(--rule);
      padding: 48px 0 38px;
    }

    .abundant-brand {
      display: table;
      margin: 0 auto 16px;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 12px;
      letter-spacing: 0.04em;
      text-decoration: none;
    }

    .abundant-brand:before {
      content: "";
      display: inline-block;
      width: 7px;
      height: 7px;
      margin-right: 8px;
      border-radius: 50%;
      background: var(--accent);
      vertical-align: 0.05em;
    }

    h1.title {
      margin: 0 0 12px;
      text-align: center;
      font-size: 48px;
      font-weight: 750;
      line-height: 1.12;
      letter-spacing: 0;
      text-wrap: balance;
    }

    .title .ital {
      color: var(--accent);
      font-style: normal;
    }

    .lede {
      max-width: 720px;
      margin: 0 auto;
      color: var(--ink-2);
      text-align: center;
      font-size: 17px;
      line-height: 1.45;
      text-wrap: pretty;
    }

    .runner-frame {
      position: relative;
      margin: 40px 0 0;
      border: 1px solid var(--rule);
      border-radius: 2px;
      background: var(--bg-2);
      overflow: hidden;
    }

    .runner-frame canvas {
      display: block;
      width: 100%;
      height: auto;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      background: var(--bg-2);
    }

    .runner-caption {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      border-top: 1px solid var(--rule);
      background: var(--bg);
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.04em;
      padding: 8px 14px 10px;
    }

    .stats-strip {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      max-width: 760px;
      margin: 28px auto 0;
      border-top: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
    }

    .stat {
      min-width: 0;
      padding: 14px 16px 13px 0;
      border-right: 1px solid var(--rule-2);
    }

    .stat + .stat {
      padding-left: 16px;
    }

    .stat:last-child {
      border-right: 0;
    }

    .stat-num {
      margin-bottom: 6px;
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      font-size: 24px;
      font-weight: 520;
      line-height: 1;
    }

    .stat-label {
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 9.5px;
      line-height: 1.3;
      text-transform: lowercase;
    }

    section {
      border-bottom: 1px solid var(--rule);
      padding: 64px 0;
    }

    .section-head {
      margin-bottom: 26px;
    }

    .section-no {
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 13px;
      text-transform: lowercase;
    }

    .section-no .dot {
      color: var(--accent);
      margin-right: 8px;
      font-size: 11px;
    }

    h2.section-title {
      margin: 4px 0 0;
      font-size: 31px;
      font-weight: 740;
      line-height: 1.14;
      letter-spacing: 0;
      text-wrap: balance;
    }

    .section-note {
      max-width: 620px;
      margin: -10px 0 30px;
      color: var(--ink-3);
      font-size: 14px;
    }

    .bench-grid {
      display: grid;
      grid-template-columns: minmax(320px, 0.82fr) minmax(420px, 1.18fr);
      border-top: 1px solid var(--rule);
      border-left: 1px solid var(--rule);
    }

    .tool-panel {
      min-width: 0;
      border-right: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      background: var(--bg);
      padding: 22px 24px;
    }

    .panel-kicker {
      margin-bottom: 12px;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    label {
      display: block;
      margin-bottom: 7px;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    textarea {
      width: 100%;
      min-height: 132px;
      resize: vertical;
      border: 1px solid var(--rule);
      border-radius: 0;
      padding: 12px;
      outline: none;
      background: var(--paper);
      color: var(--ink);
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.5;
    }

    textarea:focus,
    input:focus-visible {
      border-color: var(--ink);
      box-shadow: 0 0 0 2px color-mix(in srgb, var(--accent) 20%, transparent);
      outline: 0;
    }

    .samples {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 10px 0 20px;
    }

    .pill,
    .sample-button,
    .secondary-button {
      border: 1px solid var(--rule);
      background: transparent;
      color: var(--ink-2);
      font-family: var(--mono);
      font-size: 12px;
      cursor: pointer;
      padding: 5px 11px;
      min-height: 30px;
    }

    .pill {
      cursor: default;
    }

    .sample-button:hover,
    .secondary-button:hover {
      border-color: var(--ink);
      color: var(--ink);
      background: var(--bg-2);
    }

    .sliders {
      display: grid;
      gap: 15px;
      margin-top: 14px;
    }

    .field-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      margin-bottom: 5px;
    }

    .field-head span {
      color: var(--ink);
      font-family: var(--mono);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .actions {
      display: flex;
      gap: 10px;
      margin-top: 20px;
    }

    .primary-button,
    .secondary-button.action {
      display: inline-flex;
      justify-content: center;
      align-items: center;
      min-height: 42px;
      border: 1px solid var(--ink);
      border-radius: 0;
      padding: 0 18px;
      font-family: var(--mono);
      font-size: 13px;
      font-weight: 700;
      cursor: pointer;
    }

    .primary-button {
      flex: 1;
      color: var(--bg);
      background: var(--ink);
    }

    .primary-button:hover {
      color: var(--bg);
      background: var(--accent);
      border-color: var(--accent);
    }

    .primary-button:disabled {
      cursor: wait;
      opacity: 0.68;
    }

    .secondary-button.action {
      color: var(--ink);
      background: transparent;
    }

    .lb-chart {
      min-width: 0;
      border-right: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      background: var(--bg);
      padding: 8px 22px 18px;
    }

    .lb-chart-head {
      display: flex;
      justify-content: space-between;
      padding: 8px 0;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .lb-row {
      margin: 0 -10px;
      padding: 7px 10px;
    }

    .lb-row.highlight {
      background: var(--accent-bg);
    }

    .lb-row-top {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 5px;
    }

    .lb-id {
      display: flex;
      align-items: baseline;
      gap: 9px;
      min-width: 0;
    }

    .lb-model {
      color: var(--ink);
      font-weight: 700;
      white-space: nowrap;
    }

    .lb-agent {
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 12px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .lb-score {
      flex-shrink: 0;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }

    .lb-track {
      position: relative;
      height: 9px;
      background: var(--rule-2);
    }

    .lb-fill {
      position: absolute;
      inset: 0 auto 0 0;
      background: var(--accent);
    }

    .denoise-trace {
      margin-top: 18px;
      border: 1px solid var(--rule);
      background: var(--paper);
    }

    .denoise-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--rule);
      background: var(--bg-2);
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .trace-row {
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr) 58px;
      align-items: center;
      gap: 12px;
      min-height: 44px;
      padding: 9px 14px;
      border-bottom: 1px solid var(--rule-2);
    }

    .trace-row:last-child {
      border-bottom: 0;
    }

    .trace-label,
    .trace-score {
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .trace-text {
      min-width: 0;
      color: var(--ink);
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.35;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .trace-score {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }

    .trace-meter {
      grid-column: 2 / 4;
      height: 4px;
      background: var(--rule-2);
      position: relative;
      margin-top: -4px;
    }

    .trace-meter span {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0%;
      background: var(--accent);
      transition: width 0.16s linear;
    }

    .output-wrap {
      margin-top: 18px;
      border: 1px solid var(--rule);
      background: var(--paper);
    }

    .output-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      border-bottom: 1px solid var(--rule);
      background: var(--bg-2);
      padding: 10px 14px;
    }

    .output-head h3 {
      margin: 0;
      font-size: 14px;
    }

    .status {
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      text-align: right;
    }

    .status[data-tone="busy"] {
      color: var(--warn);
    }

    .status[data-tone="error"] {
      color: var(--error);
    }

    .output-list {
      min-height: 268px;
      max-height: 520px;
      overflow: auto;
      padding: 0;
    }

    .pass-card {
      border-bottom: 1px solid var(--rule);
      background: var(--paper);
    }

    .pass-card:last-child {
      border-bottom: 0;
    }

    .pass-card-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 14px;
      border-bottom: 1px solid var(--rule-2);
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    .pass-card pre {
      margin: 0;
      padding: 14px 16px 18px;
      overflow: visible;
      white-space: pre-wrap;
      word-break: break-word;
      color: var(--ink);
      font: 13px / 1.58 var(--mono);
    }

    .placeholder {
      color: var(--ink-3);
      padding: 16px 18px;
      font-family: var(--mono);
      font-size: 13px;
    }

    .tasks-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-top: 1px solid var(--rule);
      border-left: 1px solid var(--rule);
    }

    .task {
      min-width: 0;
      border-right: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      background: var(--bg);
      padding: 20px 20px 18px;
      cursor: pointer;
      text-align: left;
    }

    .task:hover,
    .task:focus-visible {
      background: var(--accent-bg);
      outline: 0;
    }

    .task-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 10px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
    }

    .task-title {
      margin: 0 0 8px;
      color: var(--ink);
      font-size: 20px;
      font-weight: 740;
      line-height: 1.15;
    }

    .task-desc {
      margin: 0;
      color: var(--ink-2);
      font-size: 13px;
      line-height: 1.45;
    }

    .analysis-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border-top: 1px solid var(--rule);
      border-left: 1px solid var(--rule);
    }

    .analysis-card {
      border-right: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      background: var(--paper);
      padding: 22px 24px;
    }

    .analysis-card h3 {
      margin: 0 0 14px;
      font-size: 18px;
      line-height: 1.18;
    }

    .split-row {
      display: grid;
      grid-template-columns: 92px 1fr 44px;
      align-items: center;
      gap: 12px;
      margin: 11px 0;
      font-family: var(--mono);
      font-size: 12px;
    }

    .split-track {
      height: 8px;
      background: var(--rule-2);
      position: relative;
    }

    .split-fill {
      position: absolute;
      inset: 0 auto 0 0;
      background: var(--accent);
    }

    .observations {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      border-top: 1px solid var(--rule);
      border-left: 1px solid var(--rule);
    }

    .finding {
      border-right: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      padding: 22px 24px;
    }

    .finding b {
      display: block;
      margin-bottom: 8px;
      font-size: 16px;
    }

    .finding p {
      margin: 0;
      color: var(--ink-2);
      font-size: 13px;
    }

    footer {
      background: var(--bg-2);
      padding: 38px 0 52px;
    }

    .foot-meta {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--ink-3);
      font-family: var(--mono);
      font-size: 11px;
    }

    @media (max-width: 880px) {
      .topnav-inner {
        padding: 12px 24px;
      }

      .nav-links {
        display: none;
      }

      .container {
        padding: 0 24px;
      }

      .hero {
        padding-top: 36px;
      }

      h1.title {
        font-size: 40px;
      }

      .bench-grid,
      .analysis-grid {
        grid-template-columns: 1fr;
      }

      .tasks-grid,
      .observations {
        grid-template-columns: repeat(2, 1fr);
      }

      .stats-strip {
        grid-template-columns: repeat(2, 1fr);
      }

      .stat:nth-child(2n) {
        border-right: 0;
      }
    }

    @media (max-width: 560px) {
      h1.title {
        font-size: 32px;
      }

      .runner-caption,
      .foot-meta {
        flex-direction: column;
      }

      .tasks-grid,
      .observations,
      .stats-strip {
        grid-template-columns: 1fr;
      }

      .stat,
      .stat + .stat {
        border-right: 0;
        padding-left: 0;
      }
    }
  </style>
</head>
<body>
  <nav class="topnav">
    <div class="topnav-inner">
      <a class="brand" href="#">
        <span class="brand-mark" aria-hidden="true"></span>
        <span>nanoDiffusionGPT</span>
      </a>
      <div class="nav-links" aria-label="sections">
        <a href="#leaderboard">leaderboard</a>
        <a href="#tasks">tasks</a>
        <a href="#analysis">analysis</a>
        <a href="#findings">observations</a>
      </div>
      <button class="theme-toggle" type="button" id="themeToggle" aria-label="Toggle color theme">mode</button>
    </div>
  </nav>

  <main>
    <section class="hero">
      <div class="container">
        <a class="abundant-brand" href="#leaderboard">local character checkpoint</a>
        <h1 class="title">nanoDiffusionGPT <span class="ital">Evolution</span></h1>
        <p class="lede">A compact Shakespeare character model running one local generation pass while the canvas visualizes a diffusion-style transition from noise to text.</p>

        <div class="runner-frame">
          <canvas id="tokenCanvas" width="1100" height="360" aria-label="Animated diffusion evolution"></canvas>
          <div class="runner-caption">
            <span id="checkpointPill">Last good: loading</span>
            <span id="devicePill">Device: loading</span>
          </div>
        </div>

        <div class="stats-strip" aria-label="model stats">
          <div class="stat">
            <div class="stat-num" id="paramsMetric">0.80M</div>
            <div class="stat-label">parameters</div>
          </div>
          <div class="stat">
            <div class="stat-num" id="vocabMetric">65</div>
            <div class="stat-label">character vocabulary</div>
          </div>
          <div class="stat">
            <div class="stat-num" id="blockMetric">64</div>
            <div class="stat-label">context block</div>
          </div>
          <div class="stat">
            <div class="stat-num" id="latencyMetric">--</div>
            <div class="stat-label">single pass latency</div>
          </div>
        </div>
      </div>
    </section>

    <section id="leaderboard">
      <div class="container wide">
        <div class="section-head">
          <div class="section-no"><span class="dot">●</span>01 / live run</div>
          <h2 class="section-title">Generator Leaderboard</h2>
        </div>

        <div class="bench-grid">
          <form class="tool-panel" id="generateForm">
            <div class="panel-kicker">prompt lane</div>
            <label for="prompt">Prompt</label>
            <textarea id="prompt" spellcheck="false">ROMEO:</textarea>

            <div class="samples" aria-label="sample prompts">
              <button type="button" class="sample-button" data-prompt="ROMEO:">ROMEO:</button>
              <button type="button" class="sample-button" data-prompt="JULIET:">JULIET:</button>
              <button type="button" class="sample-button" data-prompt="KING RICHARD:">KING RICHARD:</button>
              <button type="button" class="sample-button" data-prompt="HAMLET:">HAMLET:</button>
            </div>

            <div class="sliders">
              <div>
                <div class="field-head">
                  <label for="maxTokens">Tokens</label>
                  <span id="maxTokensValue">220</span>
                </div>
                <input id="maxTokens" type="range" min="40" max="500" step="10" value="220">
              </div>

              <div>
                <div class="field-head">
                  <label for="temperature">Temperature</label>
                  <span id="temperatureValue">0.80</span>
                </div>
                <input id="temperature" type="range" min="0.40" max="1.40" step="0.05" value="0.80">
              </div>

              <div>
                <div class="field-head">
                  <label for="topK">Top K</label>
                  <span id="topKValue">35</span>
                </div>
                <input id="topK" type="range" min="5" max="65" step="5" value="35">
              </div>

            </div>

            <div class="actions">
              <button type="submit" class="primary-button" id="generateButton">Run One Pass</button>
              <button type="button" class="secondary-button action" id="clearButton">Clear</button>
            </div>
          </form>

          <section class="lb-chart" aria-label="generation results">
            <div class="lb-chart-head">
              <span>runner</span>
              <span>score</span>
            </div>

            <div class="lb-row highlight">
              <div class="lb-row-top">
                <div class="lb-id">
                  <span class="lb-model">last-good checkpoint</span>
                  <span class="lb-agent" id="checkpointRow">loading</span>
                </div>
                <span class="lb-score" id="scoreMetric">ready</span>
              </div>
              <div class="lb-track"><div class="lb-fill" id="scoreFill" style="width: 42%"></div></div>
            </div>

            <div class="lb-row">
              <div class="lb-row-top">
                <div class="lb-id">
                  <span class="lb-model">diffusion pass</span>
                  <span class="lb-agent" id="samplingRow">one pass / temp 0.80 / top-k 35</span>
                </div>
                <span class="lb-score" id="tokenScore">220</span>
              </div>
              <div class="lb-track"><div class="lb-fill" id="tokenFill" style="width: 44%"></div></div>
            </div>

            <div class="denoise-trace" aria-label="Realtime denoising trace">
              <div class="denoise-head">
                <span>word correction trace</span>
                <span id="tracePhase">idle</span>
              </div>
              <div class="trace-row">
                <span class="trace-label">base noise</span>
                <span class="trace-text" id="traceNoise">sampling prior noise</span>
                <span class="trace-score" id="traceNoiseScore">0%</span>
                <div class="trace-meter"><span id="traceNoiseFill"></span></div>
              </div>
              <div class="trace-row">
                <span class="trace-label">mask fill</span>
                <span class="trace-text" id="traceMask">waiting for prompt anchors</span>
                <span class="trace-score" id="traceMaskScore">0%</span>
                <div class="trace-meter"><span id="traceMaskFill"></span></div>
              </div>
              <div class="trace-row">
                <span class="trace-label">word snap</span>
                <span class="trace-text" id="traceSnap">aligning word candidates</span>
                <span class="trace-score" id="traceSnapScore">0%</span>
                <div class="trace-meter"><span id="traceSnapFill"></span></div>
              </div>
              <div class="trace-row">
                <span class="trace-label">repair</span>
                <span class="trace-text" id="traceRepair">polishing final line</span>
                <span class="trace-score" id="traceRepairScore">0%</span>
                <div class="trace-meter"><span id="traceRepairFill"></span></div>
              </div>
            </div>

            <div class="output-wrap">
              <div class="output-head">
                <h3>Single Pass Output</h3>
                <div class="status" id="status" data-tone="idle">Ready</div>
              </div>
              <div id="output" class="output-list"><div class="placeholder">One generated pass will appear here.</div></div>
            </div>
          </section>
        </div>
      </div>
    </section>

    <section id="tasks">
      <div class="container wide">
        <div class="section-head">
          <div class="section-no"><span class="dot">●</span>02 / prompt tasks</div>
          <h2 class="section-title">Task Grid</h2>
        </div>
        <div class="tasks-grid">
          <button type="button" class="task" data-prompt="ROMEO:">
            <div class="task-head"><span>01</span><span>dialogue</span></div>
            <h3 class="task-title">Romeo</h3>
            <p class="task-desc">Compact romantic dialogue with fast line breaks.</p>
          </button>
          <button type="button" class="task" data-prompt="JULIET:">
            <div class="task-head"><span>02</span><span>dialogue</span></div>
            <h3 class="task-title">Juliet</h3>
            <p class="task-desc">Softer phrasing and character-style turn taking.</p>
          </button>
          <button type="button" class="task" data-prompt="KING RICHARD:">
            <div class="task-head"><span>03</span><span>court</span></div>
            <h3 class="task-title">King Richard</h3>
            <p class="task-desc">Longer names, court cadence, and heavier nouns.</p>
          </button>
          <button type="button" class="task" data-prompt="HAMLET:">
            <div class="task-head"><span>04</span><span>monologue</span></div>
            <h3 class="task-title">Hamlet</h3>
            <p class="task-desc">A noisier stress test for the character checkpoint.</p>
          </button>
        </div>
      </div>
    </section>

    <section id="analysis">
      <div class="container">
        <div class="section-head">
          <div class="section-no"><span class="dot">●</span>03 / analysis</div>
          <h2 class="section-title">Run Profile</h2>
        </div>
        <p class="section-note">These live controls map directly to the sampling request sent to the local model process.</p>
        <div class="analysis-grid">
          <div class="analysis-card">
            <h3>Sampling Split</h3>
            <div class="split-row"><span>tokens</span><div class="split-track"><div class="split-fill" id="analysisTokens" style="width: 44%"></div></div><b id="analysisTokensValue">220</b></div>
            <div class="split-row"><span>temp</span><div class="split-track"><div class="split-fill" id="analysisTemp" style="width: 40%"></div></div><b id="analysisTempValue">0.80</b></div>
            <div class="split-row"><span>top-k</span><div class="split-track"><div class="split-fill" id="analysisTopK" style="width: 54%"></div></div><b id="analysisTopKValue">35</b></div>
          </div>
          <div class="analysis-card">
            <h3>Checkpoint Split</h3>
            <div class="split-row"><span>val loss</span><div class="split-track"><div class="split-fill" style="width: 68%"></div></div><b>1.8737</b></div>
            <div class="split-row"><span>iters</span><div class="split-track"><div class="split-fill" style="width: 80%"></div></div><b>2000</b></div>
            <div class="split-row"><span>device</span><div class="split-track"><div class="split-fill" style="width: 75%"></div></div><b id="deviceShort">mps</b></div>
          </div>
        </div>
      </div>
    </section>

    <section id="findings">
      <div class="container">
        <div class="section-head">
          <div class="section-no"><span class="dot">●</span>04 / observations</div>
          <h2 class="section-title">Failure Modes</h2>
        </div>
        <div class="observations">
          <div class="finding">
            <b>Short training</b>
            <p>The 50-iteration checkpoint looked broken; the active checkpoint is the 2000-iteration run.</p>
          </div>
          <div class="finding">
            <b>Character model</b>
            <p>Outputs preserve Shakespeare-like shape, but invented spellings are expected at this size.</p>
          </div>
          <div class="finding">
            <b>Fast local loop</b>
            <p>MPS keeps generation interactive while the page renders one diffusion-style evolution pass.</p>
          </div>
        </div>
      </div>
    </section>
  </main>

  <footer>
    <div class="container">
      <div class="foot-meta">
        <span>nanoDiffusionGPT - local demo</span>
        <span id="footerCheckpoint">checkpoint loading</span>
      </div>
    </div>
  </footer>

  <script>
    const root = document.documentElement;
    const form = document.getElementById("generateForm");
    const promptEl = document.getElementById("prompt");
    const outputEl = document.getElementById("output");
    const statusEl = document.getElementById("status");
    const generateButton = document.getElementById("generateButton");
    const clearButton = document.getElementById("clearButton");
    const maxTokens = document.getElementById("maxTokens");
    const temperature = document.getElementById("temperature");
    const topK = document.getElementById("topK");
    const maxTokensValue = document.getElementById("maxTokensValue");
    const temperatureValue = document.getElementById("temperatureValue");
    const topKValue = document.getElementById("topKValue");
    const themeToggle = document.getElementById("themeToggle");
    const canvas = document.getElementById("tokenCanvas");
    const ctx = canvas.getContext("2d");
    const traceRows = [
      {
        text: document.getElementById("traceNoise"),
        score: document.getElementById("traceNoiseScore"),
        fill: document.getElementById("traceNoiseFill"),
        offset: 0.00,
        noise: 0.92
      },
      {
        text: document.getElementById("traceMask"),
        score: document.getElementById("traceMaskScore"),
        fill: document.getElementById("traceMaskFill"),
        offset: 0.16,
        noise: 0.64
      },
      {
        text: document.getElementById("traceSnap"),
        score: document.getElementById("traceSnapScore"),
        fill: document.getElementById("traceSnapFill"),
        offset: 0.34,
        noise: 0.38
      },
      {
        text: document.getElementById("traceRepair"),
        score: document.getElementById("traceRepairScore"),
        fill: document.getElementById("traceRepairFill"),
        offset: 0.52,
        noise: 0.16
      }
    ];
    const tracePhase = document.getElementById("tracePhase");
    const traceLexicon = [
      "thane", "crown", "grief", "mercy", "honour", "storm", "sword", "night",
      "blood", "court", "love", "dream", "lord", "queen", "voice", "heart",
      "banish", "fortune", "truth", "vow", "sleep", "raven", "music", "grave"
    ];

    let visualActive = false;
    let generatedTokens = 0;
    let lastLatency = 0;
    let evolutionStartedAt = performance.now();
    let finalTextPreview = "";
    let lastTraceTick = 0;
    let liveOutput = null;
    let liveOutputMode = "idle";

    function setTheme(mode) {
      root.dataset.theme = mode;
      localStorage.setItem("nano-marathon-theme", mode);
      themeToggle.textContent = mode === "dark" ? "dark" : "light";
    }

    function setStatus(text, tone = "idle") {
      statusEl.textContent = text;
      statusEl.dataset.tone = tone;
    }

    function setFill(id, value) {
      document.getElementById(id).style.width = `${Math.max(3, Math.min(100, value))}%`;
    }

    function syncControls() {
      const tokenValue = Number(maxTokens.value);
      const tempValue = Number(temperature.value);
      const topKValueNum = Number(topK.value);

      maxTokensValue.textContent = tokenValue;
      temperatureValue.textContent = tempValue.toFixed(2);
      topKValue.textContent = topKValueNum;
      document.getElementById("samplingRow").textContent = `one pass / temp ${tempValue.toFixed(2)} / top-k ${topKValueNum}`;
      document.getElementById("tokenScore").textContent = tokenValue;
      document.getElementById("analysisTokensValue").textContent = tokenValue;
      document.getElementById("analysisTempValue").textContent = tempValue.toFixed(2);
      document.getElementById("analysisTopKValue").textContent = topKValueNum;
      setFill("tokenFill", (tokenValue / 500) * 100);
      setFill("analysisTokens", (tokenValue / 500) * 100);
      setFill("analysisTemp", (tempValue / 1.4) * 100);
      setFill("analysisTopK", (topKValueNum / 65) * 100);
    }

    function renderPassShell() {
      const card = document.createElement("article");
      card.className = "pass-card";
      card.dataset.pass = "1";

      const head = document.createElement("div");
      head.className = "pass-card-head";
      const title = document.createElement("span");
      title.textContent = "diffusion pass 01";
      const meta = document.createElement("span");
      meta.textContent = "running";
      head.append(title, meta);

      const pre = document.createElement("pre");
      pre.textContent = "";
      card.append(head, pre);
      outputEl.append(card);
      liveOutput = { card, meta, pre };
      return liveOutput;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function pseudo(seed) {
      return Math.abs(Math.sin(seed * 12.9898) * 43758.5453) % 1;
    }

    function cleanWords(text) {
      return (text || "")
        .replace(/[^A-Za-z0-9':;-]+/g, " ")
        .trim()
        .split(/\\s+/)
        .filter(Boolean);
    }

    function noisyWord(index, tick, rowIndex) {
      const base = traceLexicon[Math.floor(pseudo(index * 19 + tick * 0.37 + rowIndex * 7) * traceLexicon.length)];
      return corruptWord(base, 0.62, index + tick + rowIndex * 31);
    }

    function corruptWord(word, amount, seed) {
      const letters = "abcdefghijklmnopqrstuvwxyz";
      const chars = String(word || "").split("");
      if (!chars.length) return noisyWord(seed, seed, 0);

      for (let i = 0; i < chars.length; i += 1) {
        const r = pseudo(seed + i * 5.17);
        if (/[A-Za-z]/.test(chars[i]) && r < amount) {
          const next = letters[Math.floor(pseudo(seed + i * 9.91) * letters.length)];
          chars[i] = /[A-Z]/.test(chars[i]) ? next.toUpperCase() : next;
        }
      }

      if (amount > 0.42 && chars.length > 4 && pseudo(seed + 99) < amount * 0.35) {
        chars.splice(Math.floor(pseudo(seed + 55) * chars.length), 1);
      }

      return chars.join("");
    }

    function traceTargetWords() {
      const fallback = `${promptEl.value} ${traceLexicon.join(" ")}`;
      const words = cleanWords(finalTextPreview || fallback);
      while (words.length < 16) {
        words.push(traceLexicon[words.length % traceLexicon.length]);
      }
      return words.slice(0, 16);
    }

    function buildTraceText(rowIndex, rowProgress, tick) {
      const words = traceTargetWords();
      if (rowProgress >= 0.98) {
        return words.join(" ");
      }
      return words.map((word, index) => {
        const reveal = clamp(rowProgress * 1.2 - index / (words.length * 1.1), 0, 1);
        const jitter = pseudo(index * 13 + rowIndex * 41 + tick * 0.19);

        if (reveal < 0.20) {
          return noisyWord(index, tick, rowIndex);
        }

        if (reveal < 0.48 && rowIndex < 2) {
          return jitter > 0.38 ? "____" : corruptWord(word, 0.72 - reveal, index + tick + rowIndex);
        }

        if (reveal < 0.78) {
          return corruptWord(word, (1 - reveal) * traceRows[rowIndex].noise, index + tick + rowIndex);
        }

        return word;
      }).join(" ");
    }

    function formatEvolvingOutput(progress, tick) {
      const prompt = promptEl.value || "\\n";
      const targetText = finalTextPreview || `${prompt} ${traceLexicon.join(" ")}`;
      const targetWords = cleanWords(targetText);
      const promptLine = prompt.endsWith(":") ? prompt : `${prompt}:`;
      const bodyWords = targetWords.length ? targetWords : traceTargetWords();
      const wordCount = Math.max(18, Math.min(42, Math.ceil(Number(maxTokens.value) / 4)));
      const lines = [promptLine];
      let line = "";

      for (let i = 0; i < wordCount; i += 1) {
        const word = bodyWords[i % bodyWords.length];
        const reveal = clamp(progress * 1.38 - i / (wordCount * 1.08), 0, 1);
        let token;

        if (reveal < 0.18) {
          token = noisyWord(i, tick, 1);
        } else if (reveal < 0.42) {
          token = pseudo(i * 7 + tick) > 0.48 ? "____" : corruptWord(word, 0.78 - reveal, i + tick);
        } else if (reveal < 0.82) {
          token = corruptWord(word, (1 - reveal) * 0.55, i + tick * 0.7);
        } else {
          token = word;
        }

        if ((line + " " + token).trim().length > 46) {
          lines.push(line.trim());
          line = token;
        } else {
          line = `${line} ${token}`;
        }
      }

      if (line.trim()) lines.push(line.trim());
      return lines.slice(0, 7).join("\\n");
    }

    function updateDenoiseTrace(progress, nowMs) {
      if (nowMs - lastTraceTick < 110) return;
      lastTraceTick = nowMs;

      const tick = Math.floor(nowMs / 110);
      tracePhase.textContent = visualActive ? "denoising" : finalTextPreview ? "corrected" : "idle";

      traceRows.forEach((row, index) => {
        const rowProgress = clamp((progress - row.offset) / 0.46, 0, 1);
        row.text.textContent = buildTraceText(index, rowProgress, tick);
        row.score.textContent = `${Math.round(rowProgress * 100)}%`;
        row.fill.style.width = `${Math.max(3, rowProgress * 100)}%`;
      });

      if (liveOutput?.pre && liveOutputMode !== "done") {
        liveOutput.pre.textContent = formatEvolvingOutput(progress, tick);
        if (liveOutputMode === "running") {
          liveOutput.meta.textContent = `${Math.round(progress * 100)}% denoising`;
        } else if (liveOutputMode === "settling") {
          liveOutput.meta.textContent = `${Math.round(progress * 100)}% correcting`;
        }
      }
    }

    function drawPixelText(text, x, y, color, size = 13) {
      ctx.fillStyle = color;
      ctx.font = `${size}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.fillText(text, x, y);
    }

    function drawVisual() {
      const w = canvas.width;
      const h = canvas.height;
      const dark = root.dataset.theme === "dark";
      const bg = dark ? "#151713" : "#f4f4f5";
      const paper = dark ? "#121312" : "#ffffff";
      const ink = dark ? "#f4f4f5" : "#18181b";
      const muted = dark ? "#a1a1aa" : "#71717a";
      const accent = dark ? "#6bcf70" : "#4caf50";
      const now = performance.now() / 1000;
      const elapsed = performance.now() - evolutionStartedAt;
      const progress = visualActive
        ? Math.min(0.72, (elapsed % 1800) / 1800)
        : finalTextPreview
          ? Math.min(1, elapsed / 1200)
          : 0.1 + Math.sin(now * 1.5) * 0.04;

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, w, h);

      ctx.fillStyle = paper;
      ctx.fillRect(32, 36, w - 64, h - 72);
      ctx.strokeStyle = dark ? "#303530" : "#e4e4e7";
      ctx.strokeRect(32, 36, w - 64, h - 72);

      drawPixelText("WORD DIFFUSION REPAIR", 56, 70, muted, 13);
      drawPixelText("x_T noise", 56, h - 54, muted, 12);
      drawPixelText("x_0 text", w - 126, h - 54, muted, 12);

      const stages = ["base", "mask", "snap", "repair", "text"];
      const stageY = 98;
      const railX = 126;
      const railW = w - 252;
      ctx.strokeStyle = dark ? "#303530" : "#e4e4e7";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(railX, stageY);
      ctx.lineTo(railX + railW, stageY);
      ctx.stroke();
      stages.forEach((stage, index) => {
        const x = railX + (railW * index) / (stages.length - 1);
        const activeStage = progress >= index / (stages.length - 1);
        ctx.fillStyle = activeStage ? accent : paper;
        ctx.strokeStyle = activeStage ? accent : muted;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(x, stageY, 8, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        drawPixelText(stage, x - 22, stageY + 28, activeStage ? ink : muted, 11);
      });

      const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?;:,.'- ";
      const target = (finalTextPreview || `${promptEl.value} generating the next character sequence`).padEnd(180, " ");
      const cols = 30;
      const rows = 5;
      const cellW = 30;
      const cellH = 28;
      const startX = 92;
      const startY = 154;
      const sweep = startX + ((elapsed / 6) % (cols * cellW));

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          const i = row * cols + col;
          const x = startX + col * cellW;
          const y = startY + row * cellH;
          const reveal = Math.max(0, Math.min(1, progress * 1.7 - i / (cols * rows * 1.5)));
          const noiseIndex = Math.abs(Math.floor(Math.sin(i * 91.7 + now * 18) * 10000)) % chars.length;
          const char = reveal > 0.55 ? target[i % target.length] : chars[noiseIndex];
          const blockAlpha = 0.08 + reveal * 0.2 + Math.max(0, 1 - Math.abs(x - sweep) / 120) * 0.18;

          ctx.fillStyle = dark
            ? `rgba(107, 207, 112, ${blockAlpha.toFixed(3)})`
            : `rgba(76, 175, 80, ${blockAlpha.toFixed(3)})`;
          ctx.fillRect(x - 5, y - 15, 22, 20);
          drawPixelText(char === "\\n" ? "↵" : char, x, y, reveal > 0.55 ? ink : accent, 14);
        }
      }

      ctx.strokeStyle = accent;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(56, h - 88);
      ctx.lineTo(56 + (w - 112) * progress, h - 88);
      ctx.stroke();

      drawPixelText(visualActive ? "DENOISING" : finalTextPreview ? "CORRECTED" : "READY", w - 176, 70, visualActive ? accent : muted, 13);
      drawPixelText(`${generatedTokens}/${Number(maxTokens.value)} tokens`, w - 176, 92, muted, 12);
      updateDenoiseTrace(progress, performance.now());

      requestAnimationFrame(drawVisual);
    }

    async function loadHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        document.getElementById("devicePill").textContent = `Device: ${data.device}`;
        document.getElementById("deviceShort").textContent = data.device;
        document.getElementById("checkpointPill").textContent = data.checkpoint_label;
        document.getElementById("checkpointRow").textContent = data.checkpoint;
        document.getElementById("footerCheckpoint").textContent = data.checkpoint;
        document.getElementById("vocabMetric").textContent = data.vocab_size;
        document.getElementById("blockMetric").textContent = data.block_size;
        document.getElementById("paramsMetric").textContent = data.params;
      } catch (err) {
        setStatus("Health check failed", "error");
      }
    }

    async function generate() {
      const tokensPerPass = Number(maxTokens.value);
      const started = performance.now();

      generateButton.disabled = true;
      visualActive = true;
      generatedTokens = 0;
      finalTextPreview = "";
      evolutionStartedAt = performance.now();
      tracePhase.textContent = "denoising";
      outputEl.innerHTML = "";
      setStatus("Running one diffusion pass", "busy");

      try {
        const shell = renderPassShell();
        liveOutputMode = "running";
        const payload = {
          prompt: promptEl.value,
          max_new_tokens: tokensPerPass,
          temperature: Number(temperature.value),
          top_k: Number(topK.value)
        };
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || "Generation failed");
        }

        const elapsedMs = Math.round(performance.now() - started);
        lastLatency = elapsedMs;
        generatedTokens = data.generated_tokens;
        finalTextPreview = data.text.replace(/\\s+/g, " ").slice(0, 180);
        evolutionStartedAt = performance.now();
        liveOutputMode = "settling";
        setTimeout(() => {
          if (liveOutput === shell) {
            shell.pre.textContent = data.text;
            shell.meta.textContent = `${data.generated_tokens} tokens / ${data.elapsed_ms} ms`;
            liveOutputMode = "done";
          }
        }, 1250);
        document.getElementById("latencyMetric").textContent = `${elapsedMs}ms`;
        document.getElementById("scoreMetric").textContent = `1 pass / ${elapsedMs}ms`;
        setFill("scoreFill", Math.max(12, 100 - elapsedMs / 80));
        setStatus(`1 pass / ${generatedTokens} tokens / ${elapsedMs} ms`);
      } catch (err) {
        outputEl.innerHTML = "";
        const card = renderPassShell();
        liveOutputMode = "done";
        card.meta.textContent = "failed";
        card.pre.textContent = err.message;
        setStatus("Error", "error");
      } finally {
        generateButton.disabled = false;
        visualActive = false;
      }
    }

    maxTokens.addEventListener("input", syncControls);
    temperature.addEventListener("input", syncControls);
    topK.addEventListener("input", syncControls);

    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        promptEl.value = button.dataset.prompt;
        promptEl.focus();
        document.getElementById("leaderboard").scrollIntoView({ block: "start" });
      });
    });

    clearButton.addEventListener("click", () => {
      generatedTokens = 0;
      finalTextPreview = "";
      liveOutput = null;
      liveOutputMode = "idle";
      evolutionStartedAt = performance.now();
      tracePhase.textContent = "idle";
      outputEl.innerHTML = '<div class="placeholder">One generated pass will appear here.</div>';
      setStatus("Ready");
      document.getElementById("latencyMetric").textContent = lastLatency ? `${lastLatency}ms` : "--";
    });

    form.addEventListener("submit", (event) => {
      event.preventDefault();
      generate();
    });

    themeToggle.addEventListener("click", () => {
      setTheme(root.dataset.theme === "dark" ? "light" : "dark");
    });

    setTheme(localStorage.getItem("nano-marathon-theme") || "light");
    syncControls();
    loadHealth();
    drawVisual();
  </script>
</body>
</html>
"""

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>nanoDiffusionGPT Playground</title>
  <style>
    :root {
      --bg: #ffffff;
      --panel: #f6f7f8;
      --ink: #141417;
      --muted: #6f7378;
      --line: #dcdfe3;
      --accent: #2f9e44;
      --error: #b4232f;
      --mono: ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--sans);
    }

    button, textarea, input { font: inherit; }

    .playground {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
    }

    .controls {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    h1 {
      margin: 0;
      font-size: 20px;
      line-height: 1.15;
      letter-spacing: 0;
    }

    .checkpoint {
      margin-top: 6px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.45;
      word-break: break-word;
    }

    label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 7px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }

    label span {
      color: var(--ink);
      font-variant-numeric: tabular-nums;
    }

    textarea {
      width: 100%;
      min-height: 132px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 0;
      padding: 12px;
      background: #ffffff;
      color: var(--ink);
      outline: none;
      font-family: var(--mono);
      font-size: 13px;
      line-height: 1.5;
    }

    textarea:focus, input:focus-visible {
      border-color: var(--ink);
      outline: 0;
    }

    .field { display: grid; gap: 6px; }

    input[type="range"] {
      width: 100%;
      accent-color: var(--accent);
    }

    .prompt-buttons {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }

    button {
      border: 1px solid var(--ink);
      border-radius: 0;
      background: #ffffff;
      color: var(--ink);
      min-height: 38px;
      padding: 0 12px;
      cursor: pointer;
      font-family: var(--mono);
      font-size: 12px;
      font-weight: 700;
    }

    button:hover {
      background: var(--ink);
      color: #ffffff;
    }

    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #ffffff;
      min-height: 44px;
    }

    button.primary:disabled {
      opacity: 0.65;
      cursor: wait;
    }

    .actions {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      margin-top: auto;
    }

    .output-shell {
      min-width: 0;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      background: #ffffff;
    }

    .output-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      min-height: 58px;
      border-bottom: 1px solid var(--line);
      padding: 0 22px;
    }

    .output-title { font-weight: 760; }

    .status {
      color: var(--muted);
      font-family: var(--mono);
      font-size: 12px;
      text-align: right;
    }

    .status.error { color: var(--error); }

    #output {
      margin: 0;
      padding: 24px;
      overflow: auto;
      white-space: pre-wrap;
      word-break: break-word;
      font: 15px / 1.65 var(--mono);
      color: var(--ink);
    }

    #output.pending { color: #52615a; }

    @media (max-width: 760px) {
      .playground { grid-template-columns: 1fr; }
      .controls {
        border-right: 0;
        border-bottom: 1px solid var(--line);
      }
      .actions { margin-top: 0; }
      #output { min-height: 46vh; }
    }
  </style>
</head>
<body>
  <main class="playground">
    <section class="controls">
      <div>
        <h1>nanoDiffusionGPT Playground</h1>
        <div class="checkpoint" id="checkpoint">loading checkpoint</div>
      </div>

      <div>
        <label for="prompt">Prompt</label>
        <textarea id="prompt" spellcheck="false">ROMEO:</textarea>
      </div>

      <div class="prompt-buttons">
        <button type="button" data-prompt="ROMEO:">ROMEO:</button>
        <button type="button" data-prompt="JULIET:">JULIET:</button>
        <button type="button" data-prompt="HAMLET:">HAMLET:</button>
        <button type="button" data-prompt="KING RICHARD:">KING RICHARD:</button>
      </div>

      <div class="field">
        <label for="tokens">Tokens <span id="tokensValue">180</span></label>
        <input id="tokens" type="range" min="40" max="500" step="10" value="180">
      </div>

      <div class="field">
        <label for="temperature">Temperature <span id="temperatureValue">0.80</span></label>
        <input id="temperature" type="range" min="0.40" max="1.40" step="0.05" value="0.80">
      </div>

      <div class="field">
        <label for="topK">Top K <span id="topKValue">35</span></label>
        <input id="topK" type="range" min="5" max="65" step="5" value="35">
      </div>

      <div class="actions">
        <button class="primary" id="run" type="button">Generate</button>
        <button id="clear" type="button">Clear</button>
      </div>
    </section>

    <section class="output-shell">
      <div class="output-head">
        <div class="output-title">Output</div>
        <div class="status" id="status">ready</div>
      </div>
      <pre id="output">Output will recursively improve here.</pre>
    </section>
  </main>

  <script>
    const promptEl = document.getElementById("prompt");
    const tokensEl = document.getElementById("tokens");
    const tempEl = document.getElementById("temperature");
    const topKEl = document.getElementById("topK");
    const outputEl = document.getElementById("output");
    const statusEl = document.getElementById("status");
    const runButton = document.getElementById("run");
    const clearButton = document.getElementById("clear");
    const lexicon = [
      "thane", "crown", "grief", "mercy", "honour", "storm", "sword", "night",
      "blood", "court", "love", "dream", "lord", "queen", "voice", "heart",
      "banish", "fortune", "truth", "vow", "sleep", "music", "grave", "fire"
    ];

    let runId = 0;
    let active = false;
    let finalText = "";
    let settleStarted = 0;
    let started = 0;
    let lastTick = 0;

    function syncControls() {
      document.getElementById("tokensValue").textContent = tokensEl.value;
      document.getElementById("temperatureValue").textContent = Number(tempEl.value).toFixed(2);
      document.getElementById("topKValue").textContent = topKEl.value;
    }

    function setStatus(text, error = false) {
      statusEl.textContent = text;
      statusEl.className = error ? "status error" : "status";
    }

    function pseudo(seed) {
      return Math.abs(Math.sin(seed * 12.9898) * 43758.5453) % 1;
    }

    function wordsFrom(text) {
      const words = String(text || "")
        .replace(/[^A-Za-z0-9':;-]+/g, " ")
        .trim()
        .split(/\s+/)
        .filter(Boolean);
      while (words.length < 48) words.push(lexicon[words.length % lexicon.length]);
      return words;
    }

    function corrupt(word, amount, seed) {
      const letters = "abcdefghijklmnopqrstuvwxyz";
      const chars = String(word || "").split("");
      if (!chars.length) return lexicon[Math.floor(pseudo(seed) * lexicon.length)];
      for (let i = 0; i < chars.length; i += 1) {
        if (/[A-Za-z]/.test(chars[i]) && pseudo(seed + i * 5.31) < amount) {
          const next = letters[Math.floor(pseudo(seed + i * 9.17) * letters.length)];
          chars[i] = /[A-Z]/.test(chars[i]) ? next.toUpperCase() : next;
        }
      }
      if (amount > 0.48 && chars.length > 4 && pseudo(seed + 77) < amount * 0.3) {
        chars.splice(Math.floor(pseudo(seed + 11) * chars.length), 1);
      }
      return chars.join("");
    }

    function draftText(progress, tick) {
      const prompt = promptEl.value || "\\n";
      const target = finalText || `${prompt} ${lexicon.join(" ")} ${lexicon.join(" ")}`;
      const words = wordsFrom(target);
      const wanted = Math.max(28, Math.min(84, Math.ceil(Number(tokensEl.value) / 3)));
      const lines = [prompt.endsWith(":") ? prompt : `${prompt}:`];
      let line = "";

      for (let i = 0; i < wanted; i += 1) {
        const word = words[i % words.length];
        const reveal = Math.max(0, Math.min(1, progress * 1.45 - i / (wanted * 1.05)));
        let token;
        if (reveal < 0.15) {
          token = corrupt(lexicon[Math.floor(pseudo(i + tick * 0.41) * lexicon.length)], 0.78, i + tick);
        } else if (reveal < 0.38) {
          token = pseudo(i * 3 + tick) > 0.46 ? "____" : corrupt(word, 0.82 - reveal, i + tick);
        } else if (reveal < 0.86) {
          token = corrupt(word, (1 - reveal) * 0.58, i + tick * 0.3);
        } else {
          token = word;
        }

        if ((line + " " + token).trim().length > 66) {
          lines.push(line.trim());
          line = token;
        } else {
          line = `${line} ${token}`;
        }
      }

      if (line.trim()) lines.push(line.trim());
      return lines.join("\\n");
    }

    function animate(id) {
      if (id !== runId) return;
      const now = performance.now();
      if (now - lastTick > 90) {
        lastTick = now;
        const tick = Math.floor(now / 90);
        let progress;
        if (active) {
          progress = Math.min(0.72, ((now - started) % 1800) / 1800);
          setStatus(`improving ${Math.round(progress * 100)}%`);
        } else if (finalText) {
          progress = Math.min(1, (now - settleStarted) / 1300);
          setStatus(progress < 1 ? `correcting ${Math.round(progress * 100)}%` : "finished");
        } else {
          progress = 0;
        }

        outputEl.textContent = progress >= 1 && finalText ? finalText : draftText(progress, tick);
        outputEl.classList.toggle("pending", progress < 1);
      }
      if (active || (finalText && performance.now() - settleStarted < 1500)) {
        requestAnimationFrame(() => animate(id));
      }
    }

    async function generate() {
      const id = ++runId;
      active = true;
      finalText = "";
      started = performance.now();
      lastTick = 0;
      outputEl.textContent = "";
      outputEl.classList.add("pending");
      runButton.disabled = true;
      setStatus("improving 0%");
      requestAnimationFrame(() => animate(id));

      try {
        const res = await fetch("/api/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            prompt: promptEl.value,
            max_new_tokens: Number(tokensEl.value),
            temperature: Number(tempEl.value),
            top_k: Number(topKEl.value)
          })
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "generation failed");

        finalText = data.text;
        active = false;
        settleStarted = performance.now();
        runButton.disabled = false;
        requestAnimationFrame(() => animate(id));
      } catch (err) {
        active = false;
        runButton.disabled = false;
        outputEl.textContent = err.message;
        outputEl.classList.remove("pending");
        setStatus("error", true);
      }
    }

    function clearOutput() {
      runId += 1;
      active = false;
      finalText = "";
      outputEl.textContent = "Output will recursively improve here.";
      outputEl.classList.remove("pending");
      setStatus("ready");
      runButton.disabled = false;
    }

    document.querySelectorAll("[data-prompt]").forEach((button) => {
      button.addEventListener("click", () => {
        promptEl.value = button.dataset.prompt;
        promptEl.focus();
      });
    });
    tokensEl.addEventListener("input", syncControls);
    tempEl.addEventListener("input", syncControls);
    topKEl.addEventListener("input", syncControls);
    runButton.addEventListener("click", generate);
    clearButton.addEventListener("click", clearOutput);

    fetch("/api/health")
      .then((res) => res.json())
      .then((data) => {
        document.getElementById("checkpoint").textContent = `${data.checkpoint} / ${data.device}`;
      })
      .catch(() => {
        document.getElementById("checkpoint").textContent = "checkpoint unavailable";
      });

    syncControls();
  </script>
</body>
</html>
"""


class DemoModel:
    def __init__(self, out_dir: Path, device: str | None):
        self.out_dir = out_dir
        self.device = device or self._default_device()
        self.lock = threading.Lock()

        checkpoint_path = out_dir / "ckpt.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        self.checkpoint_path = checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = checkpoint["config"]
        gptconf = GPTConfig(**checkpoint["model_args"])
        self.model = GPT(gptconf)
        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for key in list(state_dict.keys()):
            if key.startswith(unwanted_prefix):
                state_dict[key[len(unwanted_prefix) :]] = state_dict.pop(key)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        meta_path = ROOT / "data" / self.config["dataset"] / "meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        with meta_path.open("rb") as f:
            meta = pickle.load(f)
        self.stoi = meta["stoi"]
        self.itos = meta["itos"]
        self.vocab_size = meta["vocab_size"]
        self.block_size = self.model.config.block_size
        self.params = f"{self.model.get_num_params() / 1e6:.2f}M"

    @staticmethod
    def _default_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def encode(self, text: str) -> list[int]:
        missing = sorted({ch for ch in text if ch not in self.stoi})
        if missing:
            pretty = " ".join(repr(ch) for ch in missing[:8])
            raise ValueError(f"Prompt contains unsupported character(s): {pretty}")
        return [self.stoi[ch] for ch in text]

    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos[i] for i in tokens)

    def health(self) -> dict[str, object]:
        checkpoint_rel = self.checkpoint_path.relative_to(ROOT).as_posix()
        return {
            "checkpoint": checkpoint_rel,
            "checkpoint_label": f"Last good: {checkpoint_rel}",
            "device": self.device,
            "vocab_size": self.vocab_size,
            "block_size": self.block_size,
            "params": self.params,
        }

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
    ) -> dict[str, object]:
        prompt = prompt or "\n"
        max_new_tokens = max(1, min(max_new_tokens, 800))
        temperature = max(0.05, min(temperature, 2.0))
        top_k = max(1, min(top_k, self.vocab_size))
        start_ids = self.encode(prompt)

        x = torch.tensor(start_ids, dtype=torch.long, device=self.device)[None, ...]
        started = time.perf_counter()
        with self.lock:
            with torch.no_grad():
                y = self.model.generate(
                    x,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        return {
            "text": self.decode(y[0].tolist()),
            "generated_tokens": max_new_tokens,
            "elapsed_ms": elapsed_ms,
        }


class DemoHandler(BaseHTTPRequestHandler):
    model: DemoModel

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            self._send(HTTPStatus.OK, HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/health":
            self._send_json(HTTPStatus.OK, self.model.health())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/api/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            response = self.model.generate(
                prompt=str(payload.get("prompt", "")),
                max_new_tokens=int(payload.get("max_new_tokens", 220)),
                temperature=float(payload.get("temperature", 0.8)),
                top_k=int(payload.get("top_k", 35)),
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, response)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        self._send(
            status,
            json.dumps(payload).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the nanoDiffusionGPT web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(ROOT)
    DemoHandler.model = DemoModel(args.out_dir.resolve(), args.device)
    server = ThreadingHTTPServer((args.host, args.port), DemoHandler)
    print(f"Demo running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
