\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{abook}[2016/12/11 v1.7 Book of Abstracts]

% Copyright (c) 2016 Andrey Russyev <andrey.russev@gmail.com>
%
% Permission is hereby granted, free of charge, to any person obtaining
% a copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
%
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
% THE SOFTWARE.

\DeclareOption{ascii}{\def\abook@inputenc{ascii}}
\DeclareOption{cp1251}{\def\abook@inputenc{cp1251}}
\DeclareOption{koi8-r}{\def\abook@inputenc{koi8-r}}
\DeclareOption{koi8-u}{\def\abook@inputenc{koi8-u}}
\DeclareOption{utf-8}{\def\abook@inputenc{utf8}}
\PassOptionsToClass{10pt}{article}
\PassOptionsToClass{a5paper}{article}
\PassOptionsToClass{twoside}{article}
\DeclareOption*{\ClassError{abook}{Option `\CurrentOption' not supported}{}}
\ExecuteOptions{ascii}
\ProcessOptions
\LoadClass{article}

\IfFileExists{rerunfilecheck.sty}{
  \RequirePackage[mainaux,starttoc,index]{rerunfilecheck}}{}

\RequirePackage[T2A]{fontenc}
\RequirePackage[\abook@inputenc]{inputenc}
\RequirePackage[english,russian,ukrainian]{babel}

\RequirePackage{indentfirst}
\RequirePackage[noadjust]{cite}
\RequirePackage{enumerate}
\RequirePackage{multicol}
\RequirePackage{textcase}

\RequirePackage{amsfonts}
\RequirePackage{mathrsfs}
\RequirePackage{amssymb}
\RequirePackage{latexsym}
\RequirePackage[misc]{ifsym}

\RequirePackage{amsmath}
\RequirePackage{amsthm}
\RequirePackage{amscd}

\RequirePackage{xcolor}
\RequirePackage{graphicx}
\RequirePackage{tikz}
\RequirePackage[hidelinks]{hyperref}

\RequirePackage{longtable}
\RequirePackage[all]{xy}
%\RequirePackage{todonotes}
%\RequirePackage{blindtext}
\RequirePackage{changes}
\RequirePackage{makeidx}

% page layout

\textwidth=\paperwidth
\textheight=\paperheight
\oddsidemargin=20mm
\evensidemargin=20mm
\advance\textwidth-\oddsidemargin
\advance\textwidth-\evensidemargin
\topmargin=18mm
\headsep=3mm
\advance\topmargin-\headheight
\advance\topmargin-\headsep
\advance\textheight-18mm
\advance\textheight-19.784mm
\footskip=10mm
\advance\oddsidemargin-1in
\advance\evensidemargin-1in
\advance\topmargin-1in
\marginparwidth=18mm

\usepackage[pass]{geometry}

% headers and footers

\let\@oddhead\relax
\let\@evenhead\relax
\def\@oddfoot{\hbox to \textwidth{\hfill\normalsize\thepage\hfill}}
\def\@evenfoot{\hbox to \textwidth{\hfill\normalsize\thepage\hfill}}

% contents

\let\ltx@tableofcontents\tableofcontents
\def\tableofcontents{%
  \addtocontents{toc}{%
    \pdfbookmark[\toclevel@section]{\contentsname}{\contentsname}}%
  \ltx@tableofcontents}
\def\format@tocline#1#2{%
  \texorpdfstring{{\it#1}\hfil\null\par#2}%
  {#1.\space#2}}
\def\l@section#1#2{%
  \addpenalty\@secpenalty
  \addvspace{.75em \@plus\p@}%
  \setlength\@tempdima{1.5em}%
  \begingroup
    \parindent\z@\rightskip\@pnumwidth
    \parfillskip-\@pnumwidth
    \leavevmode
    \advance\leftskip\@tempdima
    \hskip-\leftskip
    #1\nobreak\leaders\hbox{\kern0.5pt\tiny.}\hfil\nobreak
    \hb@xt@\@pnumwidth{\hss\it\hyperpage{#2}}\par
  \endgroup}

% abstract environment, authors, title, etc.

\newcounter{abstract}
\@addtoreset{equation}{abstract}
\newif\if@samepage
\@samepagefalse
\def\abstract{%
  \if@samepage\else\clearpage\fi
  \refstepcounter{abstract}%
  \@firstauthortrue
  \def\@toc@authors{}%
  \def\@authors{}%
  \def\@addresses{}%
  \def\@emails{}%
  \usecounter{author}%
  \usecounter{address}%
  \local@preamble
  \vspace{12pt}}
\def\endabstract{%
  \end@local@preamble
  \ifx\@emails\@empty\else
    \begin{trivlist}
    \raggedright
    \it\footnotesize
    \settowidth\leftskip{E-mail:\space}%
    \item
    \leavevmode
    \hbox to 0pt{\hss E-mail:\space}%
    \MakeLowercase{\@emails}.
    \end{trivlist}
  \fi}
\newenvironment{abstract*}{\@samepagetrue\abstract}{\endabstract}
\def\title{\end@local@preamble\@dblarg\abstract@title}
\def\abstract@title[#1]#2{\gdef\@toc@title{#1}\gdef\@title{#2}}
\newif\if@firstauthor
\newif\if@mourning
\def\mourning@fbox#1#2{{\fboxsep=1.5pt\fbox{#1#2}}}
\newcounter{author}
\newcounter{address}
\def\author{\end@local@preamble\@dblarg\abstract@author}
\def\abstract@author[#1]#2#3#4{%
  \if@mourning
    \def\last@author{\texorpdfstring{\mourning@fbox{#2}}{#2}}%
  \else
    \def\last@author{#2}%
  \fi
  \def\last@address{#3}%
  \def\last@email{#4}%
  \index{\languagename:#1}%
  \addtocounter{author}{1}%
  \edef\author@index{$^{\arabic{author}}$}%
  \if@firstauthor\else\g@addto@macro\@toc@authors{, }%
    \g@addto@macro\@authors{, }\fi
  \@xp\g@addto@macro\@xp\@toc@authors\@xp{\last@author{}}%
  \@xp\g@addto@macro\@xp\@authors\@xp{\last@author}%
  \ifx\last@address\@empty
    \ifx\last@email\@empty
      \let\author@index\@empty\fi\fi
  \@xp\g@addto@macro\@xp\@authors\@xp{\@xp{\author@index}}%
  \ifx\last@address\@empty\addtocounter{address}{1}\else
    \@xp\ifx\csname\meaning\last@address\endcsname\relax
      \@xp\def\csname\meaning\last@address\endcsname{}%
      \@xp\g@addto@macro\@xp\@addresses\@xp{%
        \@xp\abstract@address@note\@xp{%
          \@xp\csname\meaning\last@address\endcsname}%
        #3\par\nobreak\smallskip}%
      \addtocounter{address}{1}%
    \fi
    \@xp\edef\csname\meaning\last@address\endcsname{%
      \csname\meaning\last@address\endcsname,%
      \arabic{author}}%
  \fi
  \ifx\last@email\@empty\else
    \ifx\@emails\@empty\else\g@addto@macro\@emails{,\space}\fi
    \g@addto@macro\@emails{\Letter\kern1.5pt\relax}%
    \@xp\g@addto@macro\@xp\@emails\@xp{\author@index}%
    \@xp\g@addto@macro\@xp\@emails\@xp{\last@email}%
  \fi
  \@firstauthorfalse
  \@mourningfalse}
\def\abstract@address@note#1{$^{\@xp\@xp\@xp\@gobble#1}$\kern2pt}
\def\mourning{\@mourningtrue}
\def\maketitle{{%
  \ifnum\c@author=1
    \let\@authors\last@author
    \let\@addresses\last@address
    \if\last@email\@empty\else
      \gdef\@emails{}\g@addto@macro\@emails{\Letter\kern3pt\relax}%
      \@xp\g@addto@macro\@xp\@emails\@xp{\last@email}\fi
  \fi
  \ifnum\c@address=1
    \def\abstract@address@note##1{}%
  \fi
  \goodbreak
  \samepage
  \begin{center}
  \@authors{}\\*[2ex]
  \bf\@title
  \addcontentsline{toc}{section}%
    {\format@tocline{\@toc@authors}{\@toc@title}}
  \end{center}
  \begin{flushleft}
  \footnotesize
  \@addresses
  \end{flushleft}
  \par}}

% local preamble

\def\save@macro#1{%
  \@xp\let\csname macro@\string#1\endcsname#1}
\def\restore@macro#1{%
  \@xp\let\@xp#1\csname macro@\string#1\endcsname}
\let\do\save@macro
\@preamblecmds
\def\local@preamble{%
  \restore@macro\@preamblecmds
  \let\do\restore@macro
  \@preamblecmds}
\def\end@preamble{
  \def\do##1{\global\let##1\@notprerr}
  \@preamblecmds}
\def\begin@local@preamble{%
  \def\end@local@preamble{\end@preamble
    \let\end@local@preamble\relax}%
  \local@preamble}
\let\end@local@preamble\relax

% lists

\def\reset@listmargins{%
  \leftmargini=0pt
  \leftmarginii=\parindent}
%\reset@listmargins
\def\labelenumii{\theenumii)}
\def\@enum@{%
  %\reset@listmargins
  \list
    {\csname label\@enumctr\endcsname}%
    {\usecounter\@enumctr}}
\def\itemize{%
  \ifnum\@itemdepth>\thr@@\@toodeep\else
    \advance\@itemdepth\@ne
    \edef\@itemitem{labelitem\romannumeral\the\@itemdepth}%
    \expandafter
    \list
      \csname\@itemitem\endcsname
      {}%
  \fi}
\def\@listI{%
  \leftmargin=\leftmargini
  \labelwidth=\leftmargini
  \advance\labelwidth-\labelsep
  \topsep=0pt
  \partopsep=0pt
  \parsep=0pt
  \itemsep=\parskip}
\def\@listii{%
  \leftmargin=\leftmarginii
  \labelwidth=\leftmarginii
  \advance\labelwidth-\labelsep
  \topsep=0pt
  \partopsep=0pt
  \parsep=0pt
  \itemsep=0pt}

% theorem environments

\def\thm@space@setup{%
  \thm@preskip=0pt
  \thm@postskip=\thm@preskip}
\def\th@plain{%
  \thm@preskip\topsep
  \divide\thm@preskip\tw@
  \thm@postskip\thm@preskip
  \itshape}
\def\th@definition{%
  \thm@preskip\topsep
  \divide\thm@preskip\tw@
  \thm@postskip\thm@preskip
  \normalfont}
\def\th@remark{%
  \thm@preskip\topsep
  \divide\thm@preskip\tw@
  \thm@postskip\thm@preskip
  \thm@headfont{\itshape}%
  \normalfont}

\theoremstyle{plain}
\newtheorem{theorem}{\theoremname}
\newtheorem*{theorem*}{\theoremname}
\newtheorem{proposition}[theorem]{\propositionname}
\newtheorem*{proposition*}{\propositionname}
\newtheorem{lemma}[theorem]{\lemmaname}
\newtheorem*{lemma*}{\lemmaname}
\newtheorem{corollary}[theorem]{\corollaryname}
\newtheorem*{corollary*}{\corollaryname}
\newtheorem{conjecture}{\conjecturename}
\newtheorem*{conjecture*}{\conjecturename}
\theoremstyle{definition}
\newtheorem{definition}{\definitionname}
\newtheorem*{definition*}{\definitionname}
\newtheorem{example}{\examplename}
\newtheorem*{example*}{\examplename}
\theoremstyle{remark}
\newtheorem{remark}{\remarkname}
\newtheorem*{remark*}{\remarkname}

\@addtoreset{theorem}{abstract}
\@addtoreset{conjecture}{abstract}
\@addtoreset{definition}{abstract}
\@addtoreset{example}{abstract}
\@addtoreset{remark}{abstract}

% bibliography

\def\@biblabel#1{[#1]}
\let\ltx@bibliography\bibliography
\def\bibliography#1{{%
  \languageshorthands{ukrainian}\mdqon
  %\bibliographystyle{gost2003m}% modified gost2003
  \bibliographystyle{gost2003}%
  \ltx@bibliography{#1}}}
\renewenvironment{thebibliography}[1]
  {\vspace{6pt}%
    %\centerline{\bfseries\itshape\small\refname}%
    %\vspace*{6pt}%
    \footnotesize
    % redefine \selectlanguage for gost2003
    \def\selectlanguage##1{\language\csname l@##1\endcsname}%
    \def\url##1{{\footnotesize##1}}
    \list{\@biblabel{\@arabic\c@enumiv}}{%
      \settowidth\labelwidth{\@biblabel{#1}}%
      \leftmargin\labelwidth
      \advance\leftmargin\labelsep
      \itemsep=0pt
      \parsep=0pt
      \@openbib@code
      \usecounter{enumiv}%
      \let\p@enumiv\@empty
      \renewcommand\theenumiv{\@arabic\c@enumiv}}%
    \sloppy
    \clubpenalty4000%
    \@clubpenalty\clubpenalty
    \widowpenalty4000%
    \sfcode`\.\@m}
  {\def\@noitemerr{\@latex@warning{Empty `thebibliography' environment}}%
    \samepage\endlist}
\def\citepunct{;\penalty4000\ }

% language dependent definitions

\let\ltx@textnumero\textnumero
\addto\captionsenglish{%
  \def\refname{References}%
  \def\theoremname{Theorem}%
  \def\propositionname{Proposition}%
  \def\lemmaname{Lemma}%
  \def\corollaryname{Corollary}%
  \def\conjecturename{Conjecture}%
  \def\definitionname{Definition}%
  \def\remarkname{Remark}%
  \def\examplename{Example}%
  \def\textnumero{No.}%
}
\addto\captionsukrainian{%
  \def\refname{\CYRL\cyrii\cyrt\cyre\cyrr\cyra\cyrt\cyru\cyrr\cyrya}%
  \def\theoremname{\CYRT\cyre\cyro\cyrr\cyre\cyrm\cyra}%
  \def\propositionname{\CYRT\cyrv\cyre\cyrr\cyrd\cyrzh\cyre\cyrn\cyrn\cyrya}%
  \def\lemmaname{\CYRL\cyre\cyrm\cyra}%
  \def\corollaryname{\CYRN\cyra\cyrs\cyrl\cyrii\cyrd\cyro\cyrk}%
  \def\conjecturename{\CYRG\cyrii\cyrp\cyro\cyrt\cyre\cyrz\cyra}%
  \def\definitionname{\CYRO\cyrz\cyrn\cyra\cyrch\cyre\cyrn\cyrn\cyrya}%
  \def\remarkname{\CYRZ\cyra\cyru\cyrv\cyra\cyrzh\cyre\cyrn\cyrn\cyrya}%
  \def\examplename{\CYRP\cyrr\cyri\cyrk\cyrl\cyra\cyrd}%
  \let\textnumero\ltx@textnumero
}
\addto\captionsrussian{%
  \def\refname{\CYRL\cyri\cyrt\cyre\cyrr\cyra\cyrt\cyru\cyrr\cyrya}%
  \def\theoremname{\CYRT\cyre\cyro\cyrr\cyre\cyrm\cyra}%
  \def\propositionname{\CYRU\cyrt\cyrv\cyre\cyrr\cyrzh%
    \cyrd\cyre\cyrn\cyri\cyre}%
  \def\lemmaname{\CYRL\cyre\cyrm\cyrm\cyra}%
  \def\corollaryname{\CYRS\cyrl\cyre\cyrd\cyrs\cyrt\cyrv\cyri\cyre}%
  \def\conjecturename{\CYRG\cyri\cyrp\cyro\cyrt\cyre\cyrz\cyra}%
  \def\definitionname{\CYRO\cyrp\cyrr\cyre\cyrd\cyre%
    \cyrl\cyre\cyrn\cyri\cyre}%
  \def\remarkname{\CYRZ\cyra\cyrm\cyre\cyrch\cyra\cyrn\cyri\cyre}%
  \def\examplename{\CYRP\cyrr\cyri\cyrm\cyre\cyrr}%
  \let\textnumero\ltx@textnumero
}

% index

\def\indexentry#1#2#3{%
  \item #2%
  \kern0.25em
  \leaders\hbox{\kern0.5pt\tiny.}\hfill
  \kern0.25em
  {\it #3}%
  \ignorespaces}
\def\indexpage#1{\hyperpage{#1}}
\def\indexlang#1#2#3{%
  \par % without \par space is inserted
  \selectlanguage{#1}%
  \vskip 0pt plus 12pt}
\def\authorindex{
  \renewcommand{\indexname}{List of Authors}
  \clearpage
  \columnsep=1cm
  \def\twocolumn{\begin{multicols}{2}}%
  \def\onecolumn{\end{multicols}}%
  \begin{theindex}
  \parskip=0pt
  \addcontentsline{toc}{section}{\indexname}%
  \@input{\jobname.ind}%
  \end{theindex}}

% misc declarations

\allowdisplaybreaks[4]

\endinput
