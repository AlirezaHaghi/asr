import { useRef, useState, DragEvent, ChangeEvent } from 'react'
import './AudioUploader.css'

interface Props {
  onFile: (file: File) => void
  selected: File | null
  active: boolean
}

const BARS = [38, 68, 52, 82, 44, 88, 58, 74, 48, 78, 63, 42, 70, 54, 84]

function Waveform({ active }: { active: boolean }) {
  return (
    <div className={`waveform${active ? ' waveform--active' : ''}`}>
      {BARS.map((h, i) => (
        <span
          key={i}
          className="waveform-bar"
          style={{ '--h': `${h}%`, '--d': `${i * 0.07}s` } as React.CSSProperties}
        />
      ))}
    </div>
  )
}

export function AudioUploader({ onFile, selected, active }: Props) {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const stop = (e: DragEvent) => { e.preventDefault(); e.stopPropagation() }

  const handleDrop = (e: DragEvent) => {
    stop(e)
    setDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) onFile(file)
  }

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
  }

  const cls = ['uploader', dragging && 'uploader--drag', active && 'uploader--active']
    .filter(Boolean).join(' ')

  return (
    <div
      className={cls}
      onDragEnter={e => { stop(e); setDragging(true) }}
      onDragLeave={e => { stop(e); setDragging(false) }}
      onDragOver={stop}
      onDrop={handleDrop}
      onClick={() => !active && inputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={e => e.key === 'Enter' && inputRef.current?.click()}
    >
      <input ref={inputRef} type="file" accept="audio/*" onChange={handleChange} style={{ display: 'none' }} />
      <Waveform active={active} />
      {selected ? (
        <div className="uploader-file">
          <span className="uploader-file__name">{selected.name}</span>
          <span className="uploader-file__size">{(selected.size / 1024 / 1024).toFixed(2)} MB</span>
        </div>
      ) : (
        <div className="uploader-prompt">
          <p className="uploader-prompt__title">Drop audio file here</p>
          <p className="uploader-prompt__sub">or click to browse · MP3, WAV, FLAC, M4A</p>
        </div>
      )}
    </div>
  )
}