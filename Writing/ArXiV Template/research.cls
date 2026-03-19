% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A new style for arxiv pre-prints
% Author: dlp@meta.com
% Author: pratyaybanerjeex@gmail.com
% Version: 1.0
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class declarations %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{research}
\LoadClassWithOptions{article}

% Layout %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[top=2.25cm, bottom=2.5cm, left=2.5cm, right=2.5cm, columnsep=0.65cm]{geometry}
\RequirePackage{microtype}
\RequirePackage{placeins}
\RequirePackage{hyphenat}
\RequirePackage{setspace}
\RequirePackage{parskip}
\RequirePackage[latin, english]{babel}
\RequirePackage{lipsum}
\RequirePackage{etoolbox}

\DisableLigatures[f]{family=sf*} 

% Graphics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{graphicx}
\RequirePackage{subcaption}

% Tables %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{booktabs}
\RequirePackage{nicematrix}
\RequirePackage{multirow}
\RequirePackage{bm}
\newcommand{\nm}[1]{#1}

% Colorful stuff %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage[most]{tcolorbox}
\RequirePackage{xcolor}
\definecolor{linkcolor}{HTML}{0064E0}
\definecolor{foregroundcolor}{HTML}{1C2B33}
\definecolor{backgroundcolor}{HTML}{DDDDFF}
\newcommand{\sectioncolor}[1]{{\bfseries\color[HTML]{000000} #1}} 

% References %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{hyperref}
\RequirePackage[noabbrev,nameinlink]{cleveref}
\hypersetup{
  colorlinks,
  linkcolor=linkcolor,
  citecolor=linkcolor,
  urlcolor=linkcolor
}

\RequirePackage[round,authoryear]{natbib}
\def\bibfont{\small}

% Swap \cite and \citet
\let\oldcite\cite
\let\cite\citep
\let\citet\oldcite

% Section and caption format %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\RequirePackage{titlesec}
\titleformat*{\paragraph}{\bfseries\sectioncolor}
\titleformat*{\section}{\Large\sffamily\bfseries\sectioncolor}
\titleformat*{\subsection}{\large\sffamily\bfseries\sectioncolor}
\titleformat*{\subsubsection}{\normalsize\sffamily\bfseries\sectioncolor}
\DeclareTextFontCommand{\textbf}{\bfseries\sffamily}

\RequirePackage{caption}
\DeclareCaptionLabelSeparator{custom}{}
\DeclareCaptionFormat{custom}{{\sffamily\textbf{#1 #2}} #3}
\captionsetup{singlelinecheck=false,format=custom,labelsep=custom,font=small}
\captionsetup[sub]{singlelinecheck=true,format=custom,labelsep=custom,font=small}

% Meta custom fonts %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% http://c.caignaert.free.fr/Install-ttf-Font.pdf

\renewcommand*\ttdefault{cmvtt}
\RequirePackage[T1]{fontenc}

% \pdfmapline{+optimistic < assets/Optimistic.ttf <T1-WGL4.enc}
% \renewcommand{\sfdefault}{optimistic}

\DeclareFontFamily{T1}{optimistic}{}
\DeclareFontShape{T1}{optimistic}{m}{n}{<-> s * [0.88] optimistic}{}
\pdfmapline{+optimistic < Optimistic.ttf <T1-WGL4.enc}
\renewcommand{\sfdefault}{optimistic}

% Authors, affiliations, infos %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\renewcommand\addtolist[5][]{
  \begingroup
    \if\relax#3\relax\def\sep{}\else\def\sep{#5}\fi
    \let\protect\@unexpandable@protect 
    \xdef#3{\expandafter{#3}\sep #4[#1]{#2}}%
  \endgroup
}

\newcommand\authorlist{}
\newcommand\authorformat[2][]{{\sffamily \bfseries #2$^{#1}$}}
\renewcommand\author[2][]{\addtolist[#1]{#2}{\authorlist}{\authorformat}{, }}

\newcommand\affiliationlist{}
\newcommand\affiliationformat[2][]{{\normalsize $^{#1}$#2}}
\newcommand\affiliation[2][]{\addtolist[#1]{#2}{\affiliationlist}{\affiliationformat}{, }}

\newcommand\contributionlist{}
\newcommand\contributionformat[2][]{{\small $^{#1}$#2}}
\newcommand\contribution[2][]{\addtolist[#1]{#2}{\contributionlist}{\contributionformat}{, }}

\newcommand\metadatalist{}
\newcommand\metadataformat[2][]{{\small {\sffamily \bfseries #1:} #2}}
\newcommand\metadata[2][]{\addtolist[#1]{#2}{\metadatalist}{\metadataformat}{\par}}

\renewcommand{\abstract}[1]{\def\abstractlist{{\color{foregroundcolor} #1}}}
\newcommand{\email}[1]{\href{mailto:#1}{\texttt{#1}}}

\renewcommand\date[1]{\metadata[Date]{#1}}
\newcommand\correspondence[1]{\metadata[Correspondence]{#1}}

\renewcommand{\title}[1]{\newcommand{\titlelist}{{\huge\sffamily\bfseries #1}}}

\newcommand{\beginappendix}{\appendix{\huge\sffamily Appendix\par}}

\newcommand{\@toptitlebar}{
  {\color{linkcolor}\hrule height 0.5pt}
  \vskip 6mm
}
\newcommand{\@bottomtitlebar}{
  \vskip 6mm
  {\color{linkcolor}\hrule height 0.5pt}
}

% Titlebox %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


\newcommand{\mymaketitle}{%
  {
    \setlength{\parskip}{0mm}
    \centering
    \nohyphens
    {
      \@toptitlebar
      \titlelist\par
      \@bottomtitlebar
    }
    \vskip 6mm
    \authorlist\par
    \vskip 3mm
    \affiliationlist\par
    \vskip 3mm
    \contributionlist\par
  }
  \vskip 6mm
  \tcbset{enhanced}
  \tcbset{left=8mm}
  \tcbset{right=8mm}
  \tcbset{top=6mm}
  \tcbset{bottom=6mm}
  % \tcbset{colback=backgroundcolor}
  \tcbset{colback=white}
  \tcbset{colframe=linkcolor}
  \tcbset{boxrule=0.5pt}
  \tcbset{before skip=0pt}
  \tcbset{grow to left by=1.5pt}
  \tcbset{grow to right by=1.5pt}
  \tcbset{arc=2.5mm}
  \begin{tcolorbox}
    \setlength{\parindent}{0cm}
    \setlength{\parskip}{0.5cm}
    %
    \abstractlist\par
    %
    \vskip 0.5cm
    %
    {
      \setlength{\parskip}{0cm}
      \ifdefempty{\metadatalist}{\vspace*{0.65cm}}{\metadatalist\par}
    }
  \end{tcolorbox}
  \tcbset{reset}
  \FloatBarrier
}

\if@twocolumn%
\renewcommand{\maketitle}{
\twocolumn[%
  \mymaketitle
  \vskip 0.38cm
]
}%
\else%
\renewcommand{\maketitle}{
  \mymaketitle
}%
\fi
