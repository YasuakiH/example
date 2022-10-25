;; -*- mode: emacs-lisp; coding: utf-8; indent-tabs-mode: nil -*-

;; --------------------------------------------------------------------
;; The MIT License (MIT)
;; Copyright (C) 2022 YasuakiH
;; 
;; Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
;; 
;; The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
;; 
;; THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
;; --------------------------------------------------------------------

;; GNU Emacs Download & Install
;; http://mirrors.syringanetworks.net/gnu/emacs/windows/emacs-27/emacs-27.2-x86_64.zip

; EmacsWiki
; http://www.emacswiki.org/
; http://www.emacswiki.org/emacs/?SiteMap
; GNU Emacs Manual
; http://www.gnu.org/software/emacs/manual/html_node/emacs/index.html

;; -----------------------------
;; Package management
;; -----------------------------

;; MELPA (tr-imeのために必要)
(package-initialize)
(customize-set-variable 'package-archives
                        '(,@package-archives
                          ("melpa" . "https://melpa.org/packages/")
			  ("org" . "https://orgmode.org/elpa/")
			  ))

;; -----------------------------
;; tr-ime
;; -----------------------------
;; Emulator of GNU Emacs IME patch for Windows (tr-ime)
;; https://github.com/trueroad/tr-emacs-ime-module
(unless (package-installed-p 'tr-ime)
  (package-refresh-contents)
  (package-install 'tr-ime))
;; tr-ime を advanced モードにする。
(tr-ime-advanced-install)
;; IM のデフォルトを IME に設定
(setq default-input-method "W32-IME")
;; IME のモードライン表示設定
(setq-default w32-ime-mode-line-state-indicator "[--]")
(setq w32-ime-mode-line-state-indicator-list '("[--]" "[あ]" "[--]"))
;; IME 初期化
(w32-ime-initialize)
;; IME 制御（yes/no などの入力の時に IME を off にする）
(wrap-function-to-control-ime 'universal-argument t nil)
(wrap-function-to-control-ime 'read-string nil nil)
(wrap-function-to-control-ime 'read-char nil nil)
(wrap-function-to-control-ime 'read-from-minibuffer nil nil)
(wrap-function-to-control-ime 'y-or-n-p nil nil)
(wrap-function-to-control-ime 'yes-or-no-p nil nil)
(wrap-function-to-control-ime 'map-y-or-n-p nil nil)
(wrap-function-to-control-ime 'register-read-with-preview nil nil)
;; IME 未確定文字のフォント
(modify-all-frames-parameters '((ime-font . "MS Gothic-14")))
;; 全バッファ IME 状態同期
;; デフォルトでは、バッファ毎に IME 状態の on/off を保持していて、 バッファを切り替えると、
;; それに応じて IME 状態も切り替わります。これを、全バッファで一つの IME 状態としたい場合には、
;; 以下のようにすればできます。
(setq w32-ime-buffer-switch-p nil)

;; ----------------------
;; Emacs の一般的な指定
;; ----------------------

(prefer-coding-system 'utf-8)

;; disable startup/splash screen and startup message
(setq inhibit-startup-message t)
(setq initial-scratch-message nil)

;; 警告音/フラッシュ
(setq visible-bell nil) ;; Emacs makes an audible ding

;; Backspace Key
(global-set-key [(control ?h)] 'delete-backward-char)

;; Help command
(global-set-key "\M-?" 'help-for-help)

;; copy clipboard when select region by mouse
(setq mouse-drag-copy-region t)

(setq-default fill-column 64)
(setq-default tab-width 4)
(setq-default line-spacing 2)

;; 自動スクロール
;; https://ayatakesi.github.io/emacs/24.5/Auto-Scrolling.html
;; (setq scroll-step            1
;;       scroll-conservatively  10000)
(setq scroll-step            1
      scroll-conservatively  1000000000)

(setq initial-major-mode 'fundamental-mode)

; undo-limit
(setq undo-outer-limit 100000000)

;; line-number and column-number are displayed on modeline
(line-number-mode t)
(column-number-mode t)

;; display filename on title bar
(setq frame-title-format "%f")

;; display-time
(setq display-time-string-forms
 '((format "%s/%s(%s)%s:%s"
		 month day dayname
		 24-hours minutes
   )))
(display-time)

;; disable tool bar
(tool-bar-mode 0)
;; disable menu bar
;; (menu-bar-mode -1)

; Scrolling without moving the point
; http://www.emacswiki.org/emacs/Scrolling
    (defun gcm-scroll-down ()
      (interactive)
      (scroll-up 1))
    (defun gcm-scroll-up ()
      (interactive)
      (scroll-down 1))
    (global-set-key [(control down)] 'gcm-scroll-down)   ;[Ctrl][↓]
    (global-set-key [(control up)]   'gcm-scroll-up)     ;[Ctrl][↑]
    (global-set-key [(control \,)]   'gcm-scroll-down)   ;[Ctrl][<]
    (global-set-key [(control \.)]   'gcm-scroll-up)     ;[Ctrl][>]

; kill-line if cursor position is top of the line
(defun kill-line-twice (&optional numlines)
  "Acts like normal kill except kills entire line if at beginning"
  (interactive "p")
  (cond ((or (= (current-column) 0)
             (> numlines 1))
         (kill-line numlines))
        (t (kill-line))))
(global-set-key "\C-k" 'kill-line-twice)

; - comment-region, uncomment-region
(global-set-key "\C-c;" 'comment-region)
(global-set-key "\C-c:" 'uncomment-region)

; - replace-string, query-replace
(global-set-key "\M-r" 'replace-string)
(global-set-key "\M-\C-r" 'query-replace)

; - toggle-read-only -
(global-set-key "\C-x\C-q" 'toggle-read-only)

; - goto-line -
(global-set-key "\M-g" 'goto-line)

; Evaluate all the Emacs Lisp expressions in the buffer
(global-set-key "\C-x\C-e" 'eval-buffer)

; disable dialogue 'File XXX is really big, really wants to open it?'
(setq large-file-warning-threshold nil)

;;; newcomment.el
;;; comment-empty-lines
;;;    If nil, comment-region does not comment out empty lines. If t, it always comments out empty lines. If eol it only comments out empty lines if comments are terminated by the end of line (i.e. comment-end is empty).
(setq comment-empty-lines t) ; 空行も含めてコメントアウトする

;;; --------------------------------
;;; conf-mode
;;; --------------------------------
(add-hook 'conf-unix-mode-hook
          (lambda()
            (setq tab-width 2)
            ))

;;; --------------------------------
;;; python-mode
;;; --------------------------------

(add-to-list 'auto-mode-alist '("\\.pyx\\'" . python-mode))

;;; --------------------------------
;;; sql-mode
;;; --------------------------------

;; ファイル *.sql のヘッダ部に次のラインを加えるとEmacsはsql-modeで開始する
;; -- -*- mode: sql; sql-product: oracle; -*-

;;; --------------------------------
;;; sql-indent
;;; ---------------------------------

;; sql-indent.el --- indentation of SQL statements
;; http://www.emacswiki.org/cgi-bin/wiki.pl?SqlInd
;(eval-after-load "sql"
;  '(load-library "sql-indent"))
;(setq sql-indent-offset 4)

;;; --------------------------------
;;; PL/SQL
;;; ---------------------------------
; plsql.el --- Programming support for PL/SQL code
; Installation
; 1. Place this file somewhere in your load path
; http://www.emacswiki.org/emacs/download/plsql.el -> site-lisp/plsql.el
; 2. Then byte-compile it.
;  M-x byte-recompile-directory -> specify a el file
; (load "plsql")

; Associate a File with a Major Mode
;; setup files ending in “.js” to open in js2-mode
;; (add-to-list 'auto-mode-alist '("\\.js\\'" . js2-mode))
; setup files ending in “.pls” to open in plsql-mode
; (add-to-list 'auto-mode-alist '("\\.pls\\'" . sql-mode))
; (add-to-list 'auto-mode-alist '("\\.pkg\\'" . sql-mode))

;; --------------------------------
;; reStructuredText (RST)
;; --------------------------------

;; Emacs support for reStructuredText. This file contains a major mode
;;   http://docutils.sourceforge.net/tools/editors/emacs/README.html
;; Emacs Support for reStructuredText
;;   http://docutils.sourceforge.net/docs/user/emacs.html
;; rst.el をロード
(require 'rst)
;; ファイル拡張子を rst-mode と対応づける
(setq auto-mode-alist
      (append '(
                ("\\.rst$" . rst-mode)
                ("\\.rest$" . rst-mode)) auto-mode-alist))
;; 背景が黒い場合はこうしないと見出しが見づらい
;; (setq frame-background-mode 'dark)
;; インデントは (TABでなく) スペース使用
(add-hook 'rst-mode-hook #'(lambda() (setq indent-tabs-mode nil)))

;; ショートカット                        動作                                           覚え方
;; -----------------------------------------------------------------------------------------------
;; C-=, C-- C-=                          自動見出しレベル設定（C-- C-=で逆順)          =で線を引く
;; (リージョン設定後)C-c C-c             選択範囲をコメントアウトする                  comment
;; (リージョン設定後)C-c C-r, C-c C-l    インデントレベルを深くする／浅くする          right, left
;; (リージョン設定後)C-c C-e, C-c C-b    すべての行を数字リスト／箇条書きにできる      enumeration, bullet
;; (リージョン設定後)C-c C-d             行ブロックを設定する                          --
;; C-c C-t                               reST内の見出し(toc; table of contents)を表示 tocの頭文字
;; (リージョン設定後)C-x r t             リージョン中の各行頭へ指定文字列挿入          region text

(require 'table)


;; -------------------------------
;; ASCIIと日本語のフォントを別々に設定する
;; -------------------------------
;; 参考) https://misohena.jp/blog/2017-09-26-symbol-font-settings-for-emacs25.html
;; デフォルトはASCII用のフォントでなければダメっぽい。
(set-face-attribute 'default nil :family "Inconsolata" :height 120)
;; ASCII以外のUnicodeコードポイント全部を一括で設定する。他国語を使用する人は細かく指定した方が良いかも。
(set-fontset-font nil '(#x80 . #x10ffff) (font-spec :family "MS Gothic"))
;; 記号をデフォルトのフォントにしない。(for Emacs 25.2)
(setq use-default-font-for-symbols nil)


;; ------------------------------
;; 以下はEmacsによる管理
;; ------------------------------

(custom-set-variables
 ;; custom-set-variables was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(Buffer-menu-name-width 48)
 '(ansi-color-faces-vector
   [default default default italic underline success warning error])
 '(ansi-color-names-vector
   ["#242424" "#e5786d" "#95e454" "#cae682" "#8ac6f2" "#333366" "#ccaa8f" "#f6f3e8"])
 '(buffers-menu-max-size nil)
 '(column-number-mode t)
 '(custom-enabled-themes '(leuven))
 '(display-time-day-and-date nil)
 '(display-time-mode t)
 '(mouse-buffer-menu-maxlen 48)
 '(package-selected-packages '(org-journal journal org-plus-contrib org))
 '(safe-local-variable-values '((sql-product . oracle)))
 '(tool-bar-mode nil))
(custom-set-faces
 ;; custom-set-faces was added by Custom.
 ;; If you edit it by hand, you could mess it up, so be careful.
 ;; Your init file should contain only one such instance.
 ;; If there is more than one, they won't work right.
 '(default ((t (:family "Consolas" :foundry "outline" :slant normal :weight normal :height 158 :width normal)))))
