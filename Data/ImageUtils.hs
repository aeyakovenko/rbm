{-|
Module      : Data.ImageUtils
Description : Image utilities for visualizing Nueral Network progress
Copyright   : (c) Anatoly Yakovenko, 2015-2016
License     : MIT
Maintainer  : aeyakovenko@gmail.com
Stability   : experimental
Portability : POSIX

This module implements some image utilities for animating training progess.
-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE TypeSynonymInstances #-}
{-# LANGUAGE MultiParamTypeClasses #-}
module Data.ImageUtils(appendGIF
                      ,writeBMP
                      ,gifCat 
                      ) where

import Control.Applicative((<|>))
import qualified Data.Matrix as M
import qualified Data.Array.Repa as R
import qualified Data.Array.Repa.IO.BMP as R
import qualified Data.Array.Repa.Algorithms.Matrix as R
import Codec.Picture.Gif as G
import Codec.Picture.Types as G
import qualified Data.ByteString as BS
import Data.Matrix(Matrix(..), U, B, I)
import qualified Data.Vector.Storable as VS
import Data.Word(Word8)

-- |append a bitmap to the gif file
-- |the generated bitmap contains B number of images
-- |each row of size I treated as a square image
appendGIF:: String -> Matrix U B I -> IO ()
appendGIF sfile mm' = do
   mm <- generateBox mm'
   let check (Left err) = error err
       check (Right a) = a
       fromDynamic (G.ImageRGB8 im) = (G.greyPalette, 10,  G.extractComponent G.PlaneRed im)
       fromDynamic _  = error "unexpected image type"
   images <- (map fromDynamic <$> check <$> G.decodeGifImages <$> (BS.readFile sfile))
         <|> (return [])
   normalized <- toImage mm
   let img = (G.greyPalette, 10, normalized)
   putStrLn $ concat ["writing image: ", sfile]
   checkE $ G.writeGifImages sfile G.LoopingForever (images ++ [img])

-- |write out a bitmap
-- |the generated bitmap contains B number of images
-- |each row of size I treated as a square image
writeBMP::String -> Matrix U B I -> IO ()
writeBMP sfile bxi = do
   image <- generateBox bxi
   ar <- R.computeUnboxedP $ R.map (\xx -> (xx,xx,xx)) image
   putStrLn $ concat ["writing image: ", sfile]
   R.writeImageToBMP sfile ar

-- |concatinate multiple gifs
gifCat :: String -> [String] -> IO ()
gifCat _ [] = return ()
gifCat f1 (f2:rest) = do 
   let fromDynamic (G.ImageRGB8 im) = (G.greyPalette, 10,  G.extractComponent G.PlaneRed im)
       fromDynamic _  = error "unexpected image type"
       check (Left err) = error err
       check (Right a) = a
       getImages sfile = do (map fromDynamic <$> check <$> G.decodeGifImages <$> (BS.readFile sfile))
                        <|> (return [])
   f1s <- getImages f1
   f2s <- getImages f2
   putStrLn $ concat ["writing image: ", f1]
   checkE $ G.writeGifImages f1 G.LoopingForever (f1s ++ f2s)
   gifCat f1 rest

-- |generate a bitmap from a square matrix
generateBox::Monad m => Matrix U B I -> m (R.Array R.U R.DIM2 Word8)
generateBox mm@(Matrix bxi) = do
   !minv <- M.fold min (read "Infinity") mm
   !maxv <- M.fold max (read "-Infinity") mm
   let
       imagewidth :: Int
       imagewidth = round $ (fromIntegral $ M.col mm)**(0.5::Double)
       batches = M.row mm
       pixels = M.col mm - 1
       toPixel xx 
         | maxv == minv = 0
         | otherwise = round $ 255 * ((xx - minv)/(maxv - minv))
       computeImage (R.Z R.:. brix R.:. bcix) =
         let  (imagenum,imagepixel) = index batches pixels brix bcix
              pos =  R.Z R.:. imagenum R.:. (imagepixel + 1)
              safeIndex m (R.Z R.:.mr R.:.mc) (R.Z R.:.r R.:. c)
                  | mr <= r || mc <= c = 0
                  | otherwise = toPixel $ m R.! (R.Z R.:.r R.:.c)
         in    safeIndex bxi (R.extent bxi) pos
       imagesperside = ceiling $ (fromIntegral $ M.row mm)**(0.5::Double)
       sh = R.Z R.:. imagewidth * imagesperside R.:. imagewidth * imagesperside
   image <- R.computeUnboxedP $ R.fromFunction sh computeImage
   return image

index :: Int -> Int -> Int -> Int -> (Int,Int)
index batches pixels rr cc = (image, pixel)
   where imagewidth = round $ (fromIntegral pixels)**(0.5::Double)
         imagesperside = ceiling $ (fromIntegral batches)**(0.5::Double)
         image = (rr `div` imagewidth) * imagesperside + (cc `div` imagewidth)
         pixel = (rr `mod` imagewidth) * imagewidth + (cc `mod` imagewidth)

checkE :: Either String t -> t
checkE (Left err) = error err
checkE (Right a) = a

toImage :: Monad m => R.Array R.U R.DIM2 Word8 -> m (G.Image G.Pixel8)
toImage img = do
   return $ G.Image (R.row $ R.extent img) (R.col $ R.extent img) $ VS.fromList $ R.toList img


