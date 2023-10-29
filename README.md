<table border="0">
  <tr>
    <td><img src="web_content/thumbnail.png" alt="image description" width="1000"></td>
    <td><h1>High-Resolution Violin Transcription using Weak Labels</h1>
      We present the <em>Multi-Stream Conformer (MUSC)</em>, a SOTA violin transcriber that converts <strong>44.1 kHz raw audio</strong> into <strong>MIDI with 5.8ms time- and 10-cent frequency-resolution</strong>, and without requiring frame-wise labels during training! <br> <br>
      <a href="https://githubtocolab.com/MTG/violin-transcription/blob/main/demo.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"/></a><br> <br>
      You can transcribe solo violin YouTube recordings using the colab demo above, and you might be pleasantly surprised by its speed!:)
      
    
</td>
  </tr>
</table>



This demo is an accompanying material to the following paper:

> [N. C. Tamer, Y. Ozer, M. Muller, X. Serra, "High-Resolution Violin Transcription using Weak Labels", in Proc. ISMIR, 2023 ](https://repositori.upf.edu/bitstream/handle/10230/58121/Tamer_ism_high.pdf?sequence=1&isAllowed=y)


## Dataset

The dataset can be found under the **dataset** folder and is comprised of three violin etude books played by 22 violinists. 
We share the open-source MIDI files aligned with the performance links. 
The filenames are structured with reconstructable links to the performances: 
```
{composer}_{catalog_number}_{performer}_{YouTube_ID}-{YouTube_start_sec}-{YouTube_end_sec}.mid
```


## MUSC architecture

The model architecture can be found under the **musc** folder. 



