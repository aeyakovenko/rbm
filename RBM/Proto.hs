module RBM.Proto where

--impl modules
import Data.Array.Repa.Algorithms.Randomish
import Data.Array.Repa
import Criterion.Main
import System.Environment(getArgs)

main :: IO ()
main = do
   let
      f (Z :. r :. c) = (fromIntegral r) * (fromIntegral c)
      sh :: DIM2
      sh = Z :. 1000 :. 1000
      f1 :: IO Double
      f1 = do
         a <- computeUnboxedP (fromFunction sh f)
         a `deepSeqArray` sumAllP a

      f2 :: IO Double
      f2 = do
         let a = computeUnboxedS $ fromFunction sh f
         a `deepSeqArray` sumAllP $ a

      f3 :: DIM2 -> Double
      f3 sh' = sumAllS $ computeUnboxedS $ fromFunction sh' f

   defaultMain [
         bench "f1" $ nfIO $ f1
      ,  bench "f2" $ nfIO $ f2
      ,  bench "f3" $ whnf f3 sh
      ]

{--

$ time ./dist/build/test/test +RTS -N4
benchmarking f1
time                 1.376 s    (1.177 s .. 1.527 s)
                     0.998 R²   (0.992 R² .. 1.000 R²)
mean                 1.388 s    (1.363 s .. 1.426 s)
std dev              33.08 ms   (271.9 as .. 35.95 ms)
variance introduced by outliers: 19% (moderately inflated)

benchmarking f2
time                 1.698 s    (1.586 s .. 1.883 s)
                     0.999 R²   (0.996 R² .. 1.000 R²)
mean                 1.634 s    (1.607 s .. 1.676 s)
std dev              37.09 ms   (0.0 s .. 39.80 ms)
variance introduced by outliers: 19% (moderately inflated)

benchmarking f3
time                 7.644 s    (7.533 s .. 7.824 s)
                     1.000 R²   (1.000 R² .. 1.000 R²)
mean                 7.763 s    (7.708 s .. 7.808 s)
std dev              70.25 ms   (0.0 s .. 77.52 ms)
variance introduced by outliers: 19% (moderately inflated)

--}
