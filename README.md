CUDA_permutations_large
=======================

next_permutation for 13! and up

NOTE: Updated code! Now at least 30% faster implementation on compute 3.5.

Two tables follow, one which shows the total GPU time for only generating each permutations of n elements, and another which generates the permutations, evaluates that permutation, and performs a reduction/scan which saves the optimal answer and the permuation associated with that answer:


Generate All Permutations of Local Array only(no Evaluation) Timing table:
---
<table>
<tr>
    <th>Total elements</th><th>Number of permutations</th><th>Tesla K20c GPU time</th>
</tr>
    <tr>
    <td> 13</td><td> 6,227,020,800 </td><td> 12.17s </td>
  </tr
  <tr>
    <td> 14</td><td> 87,178,291,200 </td><td> 188.07s </td>
</tr>
<tr>
    <td> 15</td><td> 1,307,674,368,000 </td><td> 3115.0s </td>
</tr>
</table>
  
___
Generate All Permutations of Local Array with full Evaluation of Permutation, Scan and reduction table:


<table>
  <tr>
    <th>Total elements</th><th>Num permutations x evaluation steps</th><th>Tesla K20c GPU time</th><th>Tesla K40c GPU time</th>
  </tr>
  <tr>
    <td> 13</td><td> 647,610,163,200 </td><td> 17.06s </td><td> 13.95s </td>
  </tr
  <tr>
    <td> 14</td><td> 9,763,968,614,400 </td><td> 263.1s </td><td> 216.8s </td>
</tr>
 <tr>
    <td> 15</td><td> 156,920,924,160,000 </td><td> 4332 s </td><td> NA </td>
</tr>
 </table>
 
---
NOTE: no overlocking of GPU, is running at stock 706 Mhz

No CPU times were shown due to the fact that I do not have that much free time (would take many hours even in CPU parallel).
 ____
 This is adjusted version of my CUDA implementation of the STL::next_permutation() function. Generates all n! possibilites of array in local GPU memory.
Two versions, one which only generates the permutations of the array, and the other which evaluates the generated permutation, calculates the optimal answer AND a permutation responsible for the answer, caches in GPU memory, reduces over all thread blocks, and returns the optimal answer and a respective optimal permutation to host memory.

Would be very interested in seeing Python, Java, Ruby, C# or other 'higher level' language implementation of the same function. In particular any multithreaded CPU version.

Note: for the test evaluation a super simple max-DAG test was used, which can be implemented faster than n! if one uses bitmasks for dependencies. This version is just for testing, and there are other permutation problems which do need all permutations generated for evaluations. This code will do that in very fast time for a single GPU/CPU setup.

For a given value/cost data set associated with each index it is possible that more than one permutation maps to an optimal answer. In such a case the GPU version may return a different permutation than the CPU version, but the value answer should be the same.

 
 For the earlier version see my other CUDA_next_permutation project. The full evaluation version will only work with GPU of compute capability 3.0 or higher (GTX 660 or better). Will perform better on the Tesla line(or Titan) due to the higher number of 64-bit double precision units.
 
 <script>
  (function(i,s,o,g,r,a,m){i['GoogleAnalyticsObject']=r;i[r]=i[r]||function(){
  (i[r].q=i[r].q||[]).push(arguments)},i[r].l=1*new Date();a=s.createElement(o),
  m=s.getElementsByTagName(o)[0];a.async=1;a.src=g;m.parentNode.insertBefore(a,m)
  })(window,document,'script','//www.google-analytics.com/analytics.js','ga');

  ga('create', 'UA-43459430-1', 'github.com');
  ga('send', 'pageview');

</script>
[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/b2a3438cc40be860aca12c8966a10aa6 "githalytics.com")](http://githalytics.com/OlegKonings/CUDA_permutations_large)

