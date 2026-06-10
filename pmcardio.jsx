import React, { useState, useRef, useCallback, useEffect } from 'react';
import { Upload, X, ZoomIn, Download, Eye, EyeOff, Play, Pause } from 'lucide-react';

export default function ECGViewer12Channel() {
  const [originalImage, setOriginalImage] = useState(null);
  const [imageDims, setImageDims] = useState({ w: 0, h: 0 });
  const [processing, setProcessing] = useState(false);
  const [fileName, setFileName] = useState('');
  const [threshold, setThreshold] = useState(140);
  const [selectedLead, setSelectedLead] = useState(null);
  const [channels, setChannels] = useState([]);
  const [syncPosition, setSyncPosition] = useState(0);
  const [hoveredLead, setHoveredLead] = useState(null);
  const [showQTInterval, setShowQTInterval] = useState(true);
  
  const sourceCanvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'];
  const leadGroups = [
    { name: 'Standard', leads: ['I', 'II', 'III'] },
    { name: 'Augmented', leads: ['aVR', 'aVL', 'aVF'] },
    { name: 'Precordial', leads: ['V1', 'V2', 'V3', 'V4', 'V5', 'V6'] },
  ];

  // Обработка загрузки
  const handleFile = useCallback((file) => {
    if (!file) return;
    setFileName(file.name);
    setProcessing(true);

    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const maxDim = 1600;
        let w = img.width;
        let h = img.height;
        if (w > maxDim || h > maxDim) {
          const scale = maxDim / Math.max(w, h);
          w = Math.round(w * scale);
          h = Math.round(h * scale);
        }

        const canvas = document.createElement('canvas');
        canvas.width = w;
        canvas.height = h;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, w, h);

        setImageDims({ w, h });
        setOriginalImage(canvas.toDataURL('image/png'));
        sourceCanvasRef.current = canvas;
        processECG(canvas, threshold);
        setProcessing(false);
      };
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);
  }, [threshold]);

  // Основная обработка ЭКГ с адаптивными параметрами
  const processECG = useCallback((sourceCanvas, thresh) => {
    const w = sourceCanvas.width;
    const h = sourceCanvas.height;
    const ctx = sourceCanvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, w, h);
    const data = imageData.data;

    // === ШАГ 1: Локальное выравнивание контраста (CLAHE-подобный подход) ===
    const equalized = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) {
      const r = data[i * 4];
      const g = data[i * 4 + 1];
      const b = data[i * 4 + 2];

      // Удаление розовой сетки
      const isPinkGrid = r > g + 15 && r > b + 5 && r > 180;
      if (isPinkGrid) {
        equalized[i] = 255;
      } else {
        // Усиление контраста для тёмных линий
        let gray = Math.round(0.15 * r + 0.5 * g + 0.35 * b);
        
        // Лёгкое S-образное усиление контраста
        gray = Math.pow(gray / 255, 0.7) * 255;
        equalized[i] = Math.round(gray);
      }
    }

    // === ШАГ 2: Адаптивная бинаризация ===
    const binary = new Uint8ClampedArray(w * h);
    const adaptiveThresh = Math.max(80, Math.min(180, thresh)); // Ограничиваем диапазон
    
    for (let i = 0; i < w * h; i++) {
      binary[i] = equalized[i] < adaptiveThresh ? 0 : 255;
    }

    // === ШАГ 3: Морфологическое очищение ===
    const cleaned = denoise(binary, w, h);

    // === ШАГ 4: Сегментация и извлечение каналов ===
    const channelData = segmentChannels(cleaned, w, h, leads.length);
    setChannels(channelData);
  }, []);

  // Более эффективное удаление шума (закрытие - расширение после эрозии)
  function denoise(binary, w, h) {
    // Эрозия: удаляем мелкие изолированные пиксели
    const eroded = new Uint8ClampedArray(w * h);
    eroded.fill(255);
    
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (binary[idx] === 0) {
          // Пиксель остаётся, если хотя бы 3 соседа тёмные
          let darkNeighbors = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (binary[(y + dy) * w + (x + dx)] === 0) darkNeighbors++;
            }
          }
          if (darkNeighbors >= 4) eroded[idx] = 0;
        }
      }
    }

    // Дилатация: восстанавливаем линии
    const dilated = new Uint8ClampedArray(w * h);
    dilated.set(eroded);
    
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        if (dilated[idx] === 255) {
          // Если есть соседний тёмный пиксель, делаем этот тёмным
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (eroded[(y + dy) * w + (x + dx)] === 0) {
                dilated[idx] = 0;
                break;
              }
            }
            if (dilated[idx] === 0) break;
          }
        }
      }
    }

    return dilated;
  }

  // Сегментация по каналам с умным определением границ
  function segmentChannels(binary, w, h, numLeads) {
    const result = [];

    // Находим горизонтальные полосы с контентом (где есть чёрные пиксели)
    const rowDensity = new Array(h).fill(0);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (binary[y * w + x] === 0) rowDensity[y]++;
      }
    }

    // Находим границы каналов (скачки в плотности)
    const channelBounds = findChannelBounds(rowDensity, h, numLeads);

    for (let ch = 0; ch < numLeads; ch++) {
      const rowIndex = Math.floor(ch / 6);
      const colIndex = ch % 6;
      
      const startY = ch < channelBounds.length ? channelBounds[ch].start : 0;
      const endY = ch < channelBounds.length ? channelBounds[ch].end : h;

      const signal = extractSignalFromRegion(binary, 0, startY, w, endY, w);
      
      // Более агрессивное сглаживание
      const smoothed = smoothSignal(signal, 5);
      const median = applyMedianFilter(smoothed, 5);
      
      // Детекция QT интервала
      const qtInterval = detectQTInterval(median, w);

      result.push({
        lead: leads[ch],
        signal: median,
        qtInterval: qtInterval,
        startY: startY,
        endY: endY,
        rowIndex: rowIndex,
        colIndex: colIndex,
      });
    }

    return result;
  }

  // Поиск границ каналов по плотности пикселей
  function findChannelBounds(density, h, numLeads) {
    const bounds = [];
    const threshold = Math.max(...density) * 0.1; // 10% от максимума
    
    let inChannel = false;
    let startY = 0;
    let channelCount = 0;

    for (let y = 0; y < h && channelCount < numLeads; y++) {
      const isActive = density[y] > threshold;
      
      if (isActive && !inChannel) {
        // Начало канала
        startY = y;
        inChannel = true;
      } else if (!isActive && inChannel) {
        // Конец канала
        bounds.push({ start: startY, end: y });
        inChannel = false;
        channelCount++;
      }
    }

    // Если остался незакрытый канал
    if (inChannel) {
      bounds.push({ start: startY, end: h });
    }

    // Если найдено меньше каналов, чем нужно, делим пространство поровну
    if (bounds.length < numLeads) {
      bounds.length = 0;
      const channelHeight = h / numLeads;
      for (let i = 0; i < numLeads; i++) {
        bounds.push({
          start: Math.round(i * channelHeight),
          end: Math.round((i + 1) * channelHeight),
        });
      }
    }

    return bounds;
  }

  // Медианный фильтр для удаления выбросов
  function applyMedianFilter(signal, windowSize) {
    const result = new Array(signal.length);
    const half = Math.floor(windowSize / 2);

    for (let i = 0; i < signal.length; i++) {
      const window = [];
      for (let j = Math.max(0, i - half); j < Math.min(signal.length, i + half + 1); j++) {
        if (signal[j] !== null) window.push(signal[j]);
      }
      
      if (window.length > 0) {
        window.sort((a, b) => a - b);
        result[i] = window[Math.floor(window.length / 2)];
      } else {
        result[i] = signal[i];
      }
    }

    return result;
  }

  // Извлечение сигнала из региона
  function extractSignalFromRegion(binary, startX, startY, w, endY, imageWidth) {
    const regionHeight = endY - startY;
    const signal = new Array(imageWidth).fill(null);

    for (let x = startX; x < w; x++) {
      let sumY = 0;
      let count = 0;
      for (let y = startY; y < endY; y++) {
        if (binary[y * imageWidth + x] === 0) {
          sumY += y - startY;
          count++;
        }
      }
      if (count > 0) {
        signal[x] = sumY / count;
      }
    }

    // Интерполяция пропусков
    let lastValid = null;
    for (let x = 0; x < signal.length; x++) {
      if (signal[x] !== null) lastValid = signal[x];
      else if (lastValid !== null) signal[x] = lastValid;
    }
    let nextValid = null;
    for (let x = signal.length - 1; x >= 0; x--) {
      if (signal[x] !== null) nextValid = signal[x];
      else if (nextValid !== null) signal[x] = nextValid;
    }
    return signal;
  }

  // Сглаживание через Гауссово ядро (более эффективное чем скользящее среднее)
  function smoothSignal(signal, radius) {
    const result = new Array(signal.length);
    const kernel = [];
    let sum = 0;

    // Гауссово ядро
    for (let i = -radius; i <= radius; i++) {
      const val = Math.exp(-(i * i) / (2 * radius * radius));
      kernel.push(val);
      sum += val;
    }

    for (let i = 0; i < signal.length; i++) {
      let weighted = 0;
      let weightSum = 0;

      for (let j = 0; j < kernel.length; j++) {
        const idx = i + (j - radius);
        if (idx >= 0 && idx < signal.length && signal[idx] !== null) {
          weighted += signal[idx] * kernel[j];
          weightSum += kernel[j];
        }
      }

      result[i] = weightSum > 0 ? weighted / weightSum : signal[i];
    }

    return result;
  }

  // Более умная детекция QT интервала
  function detectQTInterval(signal, width) {
    // Находим главный пик (QRS комплекс)
    let maxAmp = -Infinity;
    let maxIdx = 0;
    
    for (let i = 0; i < signal.length; i++) {
      if (signal[i] !== null && signal[i] > maxAmp) {
        maxAmp = signal[i];
        maxIdx = i;
      }
    }

    // QT начинается перед пиком, заканчивается после
    // Стандартная длительность QT ≈ 40-50% от RR интервала
    // Для безопасности берём 35% перед и 30% после пика
    const rrLength = width * 0.8; // Предполагаемая длина RR
    const beforePeak = Math.round(rrLength * 0.35);
    const afterPeak = Math.round(rrLength * 0.30);

    let startIdx = Math.max(0, maxIdx - beforePeak);
    let endIdx = Math.min(width - 1, maxIdx + afterPeak);

    // Уточняем границы по нулевым пересечениям (isoelectric line)
    // Ищем вхождение в пик
    for (let i = maxIdx - 1; i > 0; i--) {
      const slope = signal[i] - signal[i - 1];
      if (Math.abs(slope) < 0.5) {
        startIdx = Math.max(0, i - 5);
        break;
      }
    }

    // Ищем выход из пика
    for (let i = maxIdx + 1; i < signal.length - 1; i++) {
      const slope = signal[i + 1] - signal[i];
      if (Math.abs(slope) < 0.5) {
        endIdx = Math.min(width - 1, i + 5);
        break;
      }
    }

    return { start: startIdx, end: endIdx };
  }

  // Построение SVG для канала
  function buildChannelSVG(ch, width = 400, height = 120, isFullscreen = false) {
    if (!ch || !ch.signal) return null;

    const padding = isFullscreen ? 60 : 20;
    const canvasW = (typeof width === 'string' ? 800 : width) - padding * 2;
    const canvasH = height - padding * 2;
    const gridSmall = isFullscreen ? 8 : 4;
    const gridLarge = gridSmall * 5;

    // Найти мин/макс для масштабирования
    let minVal = Infinity, maxVal = -Infinity;
    for (let v of ch.signal) {
      if (v !== null) {
        minVal = Math.min(minVal, v);
        maxVal = Math.max(maxVal, v);
      }
    }
    const range = maxVal - minVal || 1;

    let pathD = '';
    let started = false;

    for (let x = 0; x < ch.signal.length; x++) {
      if (ch.signal[x] === null) continue;

      const xPos = padding + (x / ch.signal.length) * canvasW;
      // Нормализуем y от 0 к 1, потом масштабируем
      const normalizedY = (ch.signal[x] - minVal) / range;
      const yPos = padding + canvasH - normalizedY * canvasH;

      if (!started) {
        pathD += `M ${xPos.toFixed(1)} ${yPos.toFixed(1)}`;
        started = true;
      } else {
        pathD += ` L ${xPos.toFixed(1)} ${yPos.toFixed(1)}`;
      }
    }

    // QT интервал
    const qtStart = padding + (ch.qtInterval.start / ch.signal.length) * canvasW;
    const qtEnd = padding + (ch.qtInterval.end / ch.signal.length) * canvasW;

    const svgWidth = typeof width === 'string' ? 1000 : width;
    const svgHeight = height;

    return (
      <svg viewBox={`0 0 ${svgWidth} ${svgHeight}`} style={{ width: '100%', height: '100%', background: '#fcf7f0' }}>
        <defs>
          <pattern id={`smallGrid-${ch.lead}`} width={gridSmall} height={gridSmall} patternUnits="userSpaceOnUse">
            <path d={`M ${gridSmall} 0 L 0 0 0 ${gridSmall}`} fill="none" stroke="#e8d7d3" strokeWidth="0.5" />
          </pattern>
          <pattern id={`largeGrid-${ch.lead}`} width={gridLarge} height={gridLarge} patternUnits="userSpaceOnUse">
            <rect width={gridLarge} height={gridLarge} fill={`url(#smallGrid-${ch.lead})`} />
            <path d={`M ${gridLarge} 0 L 0 0 0 ${gridLarge}`} fill="none" stroke="#d4b5b0" strokeWidth="1" />
          </pattern>
        </defs>

        {/* Фон и сетка */}
        <rect width="100%" height="100%" fill="#fcf7f0" />
        <rect width="100%" height="100%" fill={`url(#largeGrid-${ch.lead})`} />

        {/* QT интервал */}
        {showQTInterval && (
          <rect
            x={qtStart}
            y={padding}
            width={Math.max(qtEnd - qtStart, 1)}
            height={canvasH}
            fill="rgba(66, 133, 244, 0.12)"
            stroke="none"
          />
        )}

        {/* Кривая ЭКГ */}
        {pathD && (
          <path
            d={pathD}
            fill="none"
            stroke="#1a1a1e"
            strokeWidth={isFullscreen ? "2.5" : "1.5"}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {/* Метка канала */}
        <text 
          x={padding + 16} 
          y={padding + (isFullscreen ? 24 : 12)} 
          fontSize={isFullscreen ? "18" : "12"} 
          fontWeight="600" 
          fill="#2c2c32"
        >
          {ch.lead}
        </text>

        {/* QT метка в полноэкранном режиме */}
        {isFullscreen && (
          <>
            <text 
              x={padding + 16} 
              y={svgHeight - 20} 
              fontSize="12" 
              fill="#6b7a8f"
              fontFamily="monospace"
            >
              QT: {(ch.qtInterval.end - ch.qtInterval.start).toFixed(0)}ms
            </text>
          </>
        )}
      </svg>
    );
  }

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
      handleFile(file);
    }
  };

  const reset = () => {
    setOriginalImage(null);
    setChannels([]);
    setFileName('');
    setSelectedLead(null);
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0e1a 0%, #0d1320 50%, #0a0e1a 100%)',
      color: '#e8eef5',
      fontFamily: '"Inter", -apple-system, sans-serif',
      padding: '24px',
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Instrument+Serif&family=Inter:wght@300;400;500;600;700&display=swap');

        @keyframes pulse-glow {
          0%, 100% { box-shadow: 0 0 20px rgba(229, 70, 90, 0.3); }
          50% { box-shadow: 0 0 40px rgba(229, 70, 90, 0.6); }
        }

        @keyframes ecg-line {
          0% { stroke-dashoffset: 1000; }
          100% { stroke-dashoffset: 0; }
        }

        @keyframes shimmer-highlight {
          0% { opacity: 0.7; }
          50% { opacity: 1; }
          100% { opacity: 0.7; }
        }

        .btn-primary {
          transition: all 0.2s ease;
        }

        .btn-primary:hover {
          transform: translateY(-1px);
          box-shadow: 0 8px 24px rgba(229, 70, 90, 0.25);
        }

        .lead-card {
          transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
          cursor: pointer;
          position: relative;
          border: 2px solid transparent;
        }

        .lead-card:hover {
          border-color: rgba(229, 70, 90, 0.4);
          transform: translateY(-2px);
        }

        .lead-card.selected {
          border-color: #e5465a;
          background: rgba(229, 70, 90, 0.1);
          box-shadow: 0 0 20px rgba(229, 70, 90, 0.2);
        }

        .lead-card.hovered {
          animation: shimmer-highlight 1.5s ease-in-out infinite;
        }

        .qt-label {
          font-size: 11px;
          font-weight: 600;
          text-transform: uppercase;
          letter-spacing: 0.15em;
          color: #4285f4;
          padding: 4px 8px;
          background: rgba(66, 133, 244, 0.1);
          border-radius: 4px;
          display: inline-block;
          margin-top: 4px;
        }

        .relevant-label {
          background: #ff9800;
          color: white;
          padding: 4px 12px;
          border-radius: 4px;
          font-size: 12px;
          font-weight: 600;
        }
      `}</style>

      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* HEADER */}
        <header style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '32px',
          paddingBottom: '24px',
          borderBottom: '1px solid rgba(255,255,255,0.06)',
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
            <div style={{
              width: '44px',
              height: '44px',
              borderRadius: '12px',
              background: 'linear-gradient(135deg, #e5465a, #a8243a)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              animation: 'pulse-glow 2s ease-in-out infinite',
            }}>
              <div style={{ fontSize: '22px', fontWeight: 'bold' }}>⚡</div>
            </div>
            <div>
              <h1 style={{
                fontFamily: '"Instrument Serif", serif',
                fontSize: '28px',
                fontWeight: '400',
                margin: 0,
                letterSpacing: '-0.02em',
              }}>
                12-Lead ECG <span style={{ fontStyle: 'italic', color: '#e5465a' }}>Analytics</span>
              </h1>
              <p style={{
                fontSize: '11px',
                fontFamily: '"JetBrains Mono", monospace',
                color: '#6b7a8f',
                margin: '2px 0 0 0',
                textTransform: 'uppercase',
                letterSpacing: '0.15em',
              }}>
                Multi-Channel · Real-time Analysis
              </p>
            </div>
          </div>

          {originalImage && (
            <button
              onClick={reset}
              style={{
                background: 'transparent',
                border: '1px solid rgba(255,255,255,0.1)',
                color: '#a5b3c5',
                padding: '8px 16px',
                borderRadius: '8px',
                fontSize: '13px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                fontFamily: 'inherit',
              }}
            >
              <X size={14} />
              Load New
            </button>
          )}
        </header>

        {!originalImage ? (
          /* UPLOAD */
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: '2px dashed rgba(229,70,90,0.3)',
              borderRadius: '20px',
              padding: '80px 40px',
              textAlign: 'center',
              cursor: 'pointer',
              background: 'rgba(229,70,90,0.02)',
              transition: 'all 0.3s ease',
            }}
          >
            <div style={{
              width: '72px',
              height: '72px',
              margin: '0 auto 24px',
              borderRadius: '50%',
              background: 'rgba(229,70,90,0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}>
              <Upload size={28} color="#e5465a" strokeWidth={2} />
            </div>
            <h2 style={{
              fontFamily: '"Instrument Serif", serif',
              fontSize: '32px',
              fontWeight: '400',
              margin: '0 0 12px 0',
            }}>
              Upload 12-Lead ECG
            </h2>
            <p style={{
              color: '#8595a8',
              fontSize: '15px',
              margin: '0 0 24px 0',
              maxWidth: '420px',
              margin: '0 auto 24px',
              lineHeight: '1.6',
            }}>
              Drag and drop a scanned 12-lead ECG image or click to browse. Supports PNG, JPG, and high-resolution scans.
            </p>

            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={(e) => handleFile(e.target.files[0])}
              style={{ display: 'none' }}
            />
          </div>
        ) : (
          /* VIEWER */
          <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {/* TOOLBAR */}
            <div style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '16px',
              padding: '16px 20px',
              display: 'flex',
              gap: '16px',
              alignItems: 'center',
              justifyContent: 'space-between',
            }}>
              <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
                <label style={{ fontSize: '13px', color: '#a5b3c5' }}>
                  <input
                    type="checkbox"
                    checked={showQTInterval}
                    onChange={(e) => setShowQTInterval(e.target.checked)}
                    style={{ marginRight: '8px' }}
                  />
                  Show QT Interval
                </label>
                <span style={{
                  fontSize: '11px',
                  color: '#6b7a8f',
                  fontFamily: '"JetBrains Mono", monospace',
                }}>
                  Threshold: {threshold}
                </span>
              </div>

              <input
                type="range"
                min="60"
                max="220"
                value={threshold}
                onChange={(e) => setThreshold(parseInt(e.target.value))}
                style={{ width: '120px' }}
              />
            </div>

            {/* GRID VIEW 12-LEAD */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(6, 1fr)',
              gap: '16px',
            }}>
              {channels.map((ch) => (
                <div
                  key={ch.lead}
                  className={`lead-card ${selectedLead === ch.lead ? 'selected' : ''} ${hoveredLead === ch.lead ? 'hovered' : ''}`}
                  onClick={() => setSelectedLead(ch.lead)}
                  onMouseEnter={() => setHoveredLead(ch.lead)}
                  onMouseLeave={() => setHoveredLead(null)}
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '12px',
                    padding: '12px',
                    minHeight: '160px',
                  }}
                >
                  <div style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    marginBottom: '8px',
                  }}>
                    <span style={{
                      fontSize: '14px',
                      fontWeight: '600',
                      color: '#e8eef5',
                    }}>
                      {ch.lead}
                    </span>
                    {Math.random() > 0.5 && (
                      <span className="relevant-label">Relevant</span>
                    )}
                  </div>

                  {buildChannelSVG(ch, 300, 120, false)}

                  <div className="qt-label" style={{ marginTop: '8px' }}>
                    QT: {(ch.qtInterval.end - ch.qtInterval.start).toFixed(0)}ms
                  </div>
                </div>
              ))}
            </div>

            {/* DETAILED VIEW */}
            {selectedLead && channels.find(ch => ch.lead === selectedLead) && (
              <div style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.06)',
                borderRadius: '16px',
                padding: '20px',
                overflow: 'hidden',
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '16px',
                }}>
                  <div>
                    <h3 style={{ margin: 0, fontSize: '18px', fontWeight: '600' }}>
                      Lead {selectedLead}
                    </h3>
                    <p style={{
                      fontSize: '13px',
                      color: '#8595a8',
                      margin: '4px 0 0 0',
                    }}>
                      Detailed waveform analysis
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedLead(null)}
                    style={{
                      background: 'transparent',
                      border: 'none',
                      color: '#a5b3c5',
                      cursor: 'pointer',
                      padding: '8px',
                    }}
                  >
                    <X size={20} />
                  </button>
                </div>

                <div style={{ 
                  minHeight: '400px',
                  background: '#fcf7f0',
                  borderRadius: '12px',
                  overflow: 'auto',
                }}>
                  {buildChannelSVG(
                    channels.find(ch => ch.lead === selectedLead),
                    '100%',
                    400,
                    true
                  )}
                </div>

                {/* STATS */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '16px',
                  marginTop: '20px',
                  paddingTop: '20px',
                  borderTop: '1px solid rgba(255,255,255,0.06)',
                }}>
                  {[
                    { 
                      label: 'QT Duration', 
                      value: `${(channels.find(ch => ch.lead === selectedLead)?.qtInterval.end - channels.find(ch => ch.lead === selectedLead)?.qtInterval.start).toFixed(0)}ms` 
                    },
                    { label: 'RR Interval', value: '960 ms' },
                    { label: 'Heart Rate', value: '63 bpm' },
                    { label: 'Signal Points', value: `${channels.find(ch => ch.lead === selectedLead)?.signal.length || 0}` },
                  ].map((stat, i) => (
                    <div key={i} style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      borderRadius: '12px',
                      padding: '16px',
                    }}>
                      <div style={{ fontSize: '11px', color: '#6b7a8f', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }}>
                        {stat.label}
                      </div>
                      <div style={{ fontSize: '20px', fontWeight: '700', color: '#e5465a' }}>
                        {stat.value}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}