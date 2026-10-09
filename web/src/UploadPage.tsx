/**
 * 上传页：一个拖拽区 + 一句说明。
 *
 * 刻意只有一个动作 —— 用户打开就能看懂要干什么。
 * 拖拽与点击选文件走同一条路径。
 */

import { useRef, useState } from "react";

export function UploadPage({
  onPick,
  busy,
}: {
  onPick: (file: File) => void;
  busy?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejected, setRejected] = useState("");

  const accept = (file: File | undefined) => {
    if (!file) return;
    const ok = /\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name);
    if (!ok) {
      setRejected(`只支持 mp4 / mov / webm 视频，收到的是「${file.name}」`);
      return;
    }
    setRejected("");
    onPick(file);
  };

  return (
    <div className="up">
      <div
        className={`up__zone${dragging ? " is-dragging" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(event) => {
          event.preventDefault();
          setDragging(false);
          accept(event.dataTransfer.files?.[0]);
        }}
        onClick={() => inputRef.current?.click()}
      >
        <div className="up__icon">🎬</div>
        <div className="up__title">把爆款视频拖到这里</div>
        <div className="up__sub">或者点击从文件夹里选一个</div>
        <div className="up__hint">mp4 / mov，3 分钟以内，500MB 以内</div>
        <input
          ref={inputRef}
          type="file"
          accept="video/mp4,video/quicktime,video/webm"
          hidden
          onChange={(event) => accept(event.target.files?.[0] ?? undefined)}
        />
        {busy && <div className="up__busy">正在上传…</div>}
      </div>
      {rejected && <div className="up__error">{rejected}</div>}
    </div>
  );
}
